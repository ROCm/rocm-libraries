
#include "header.h"

void predict_unit3(union Entry* data, double* result) {
  unsigned int tmp;
  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
    result[0] += 0.0008895603757406341;
  } else {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.158509254455567294) ) ) {
      if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
        if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)4.125962495803833896) ) ) {
          result[0] += 0.00349857518331787;
        } else {
          result[0] += -0.03726203419476637;
        }
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.662244915962219682) ) ) {
          result[0] += -0.018759956634585042;
        } else {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.329314231872559482) ) ) {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += -0.05388836309711931;
                } else {
                  result[0] += -0.004870167907327771;
                }
              } else {
                result[0] += -0.009479236853198127;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.58491539955139249) ) ) {
                result[0] += -0.023691122075593277;
              } else {
                result[0] += 0.004575061456459416;
              }
            }
          } else {
            result[0] += 0.00136947590505208;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
        if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += -0.008220281384212439;
            } else {
              result[0] += -0.038747527207284846;
            }
          } else {
            if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.321723937988282138) ) ) {
                if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.018451256304113064;
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.465643882751465732) ) ) {
                      result[0] += -0.025567328009738412;
                    } else {
                      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                        result[0] += 0.008473253554317817;
                      } else {
                        result[0] += -0.014067313628415422;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.53326439857482999) ) ) {
                    result[0] += 0.0060706731073236;
                  } else {
                    if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.0587837018778975;
                    } else {
                      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                        result[0] += -0.02586044545667056;
                      } else {
                        result[0] += 0.012521361897822478;
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.04460102858086834;
                  } else {
                    result[0] += 0.04365789692603203;
                  }
                } else {
                  result[0] += -0.06377049835572116;
                }
              }
            } else {
              if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.602003335952759233) ) ) {
                    result[0] += -0.020021634261242982;
                  } else {
                    result[0] += 0.012455800257300179;
                  }
                } else {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.835998296737671787) ) ) {
                      if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += -0.01651714146031596;
                      } else {
                        result[0] += -0.0802357080468811;
                      }
                    } else {
                      if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
                          result[0] += -0.03374452931355367;
                        } else {
                          result[0] += 0.013345352912152768;
                        }
                      } else {
                        result[0] += 0.02869064430549581;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += -0.030789483454811018;
                    } else {
                      result[0] += 0.03945239455952448;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                  result[0] += 0.009622088241679348;
                } else {
                  result[0] += 0.03392493992109951;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
            result[0] += -0.0022889129683236666;
          } else {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += -0.06961034195648731;
            } else {
              result[0] += -0.021281058043440312;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
          if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
            if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.801954269409180576) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.14301252365112482) ) ) {
                    if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.938867926597595659) ) ) {
                        result[0] += 0.10957067245518388;
                      } else {
                        result[0] += 0.01781057671823399;
                      }
                    } else {
                      result[0] += -0.0013705985320231744;
                    }
                  } else {
                    if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.938867926597595659) ) ) {
                        result[0] += 0.06321350847641077;
                      } else {
                        result[0] += -0.0024103387483039333;
                      }
                    } else {
                      result[0] += -0.022260885256001058;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                    if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                      result[0] += -0.017478303864848853;
                    } else {
                      result[0] += 0.0020011896064672254;
                    }
                  } else {
                    result[0] += -0.02266232948419701;
                  }
                }
              } else {
                if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.714269638061524326) ) ) {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.918272972106934482) ) ) {
                      if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.388237953186036044) ) ) {
                        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                          result[0] += 0.0006594697941748414;
                        } else {
                          result[0] += 0.03174938215354491;
                        }
                      } else {
                        result[0] += -0.047890265256145294;
                      }
                    } else {
                      if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                        result[0] += 0.0016847948543143967;
                      } else {
                        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                          result[0] += -0.011493858796703733;
                        } else {
                          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                            result[0] += -0.060667650157934676;
                          } else {
                            result[0] += -0.0015597682434816244;
                          }
                        }
                      }
                    }
                  } else {
                    result[0] += 0.07260191021114669;
                  }
                } else {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += -0.013948172286600594;
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.329314231872559482) ) ) {
                      result[0] += 0.011565945772448766;
                    } else {
                      if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
                        result[0] += 0.021281346323439716;
                      } else {
                        result[0] += 0.06924409287394723;
                      }
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.53326439857482999) ) ) {
                result[0] += 0.0008446623458501293;
              } else {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += -0.019464399318809403;
                } else {
                  result[0] += 0.037858155589790096;
                }
              }
            }
          } else {
            result[0] += -0.028805241812492995;
          }
        } else {
          result[0] += -0.042239488751933944;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.174569487571716753) ) ) {
      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.426736354827881748) ) ) {
        if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
            if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)9.96439266204834162) ) ) {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.15100884437561124) ) ) {
                result[0] += -0.0018016290870542292;
              } else {
                if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.698768615722657138) ) ) {
                  result[0] += -0.02858859917599323;
                } else {
                  result[0] += -0.2088900402844454;
                }
              }
            } else {
              result[0] += -0.14849367186529402;
            }
          } else {
            if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += 0.0044014302512748625;
            } else {
              if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += 0.05064846740759036;
              } else {
                result[0] += -0.0267091290022635;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
            result[0] += -0.009975280057094471;
          } else {
            if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.01815199939608138;
                } else {
                  result[0] += -0.0012358760784339305;
                }
              } else {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.285887241363526279) ) ) {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                    result[0] += 0.03409552658485883;
                  } else {
                    if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)5.525167226791382724) ) ) {
                      result[0] += -0.06801471600136504;
                    } else {
                      result[0] += 0.11308539033323506;
                    }
                  }
                } else {
                  result[0] += 0.08810377328811711;
                }
              }
            } else {
              if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.349750161170959917) ) ) {
                if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)12.00000000000000178) ) ) {
                    if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.324458837509156162) ) ) {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.153024196624756748) ) ) {
                        result[0] += 0.0012703041666314095;
                      } else {
                        result[0] += -0.006592029670549823;
                      }
                    } else {
                      result[0] += 0.008019242329447694;
                    }
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.591613531112671787) ) ) {
                      if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)0.8958797454833985485) ) ) {
                        if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.071567356586456743) ) ) {
                          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.311204195022583896) ) ) {
                            result[0] += 0.009879995451779741;
                          } else {
                            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.464467763900757724) ) ) {
                              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.422362327575684482) ) ) {
                                result[0] += -0.016641780743350758;
                              } else {
                                result[0] += -0.13347396872234826;
                              }
                            } else {
                              result[0] += 0.006409677930865137;
                            }
                          }
                        } else {
                          result[0] += -0.09558302482236816;
                        }
                      } else {
                        result[0] += 0.08413178039598312;
                      }
                    } else {
                      result[0] += 0.048536865041349166;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)6.360871315002442294) ) ) {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
                      result[0] += -0.00977591588664809;
                    } else {
                      result[0] += -0.037646882057383424;
                    }
                  } else {
                    result[0] += 0.0326989545239669;
                  }
                }
              } else {
                result[0] += 0.005438318593905056;
              }
            }
          }
        }
      } else {
        result[0] += -0.008337277798104234;
      }
    } else {
      result[0] += -0.006462760793692903;
    }
  } else {
    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.553712725639343706) ) ) {
      result[0] += -0.007319842332743329;
    } else {
      if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)4.994855403900147373) ) ) {
        if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)6.197461366653443271) ) ) {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.868834793567657693) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.851041555404663974) ) ) {
              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.009235262995950516;
                } else {
                  result[0] += -0.045532133994645875;
                }
              } else {
                if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.0036139955117660293;
                } else {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)1.497866153717041238) ) ) {
                    result[0] += 0.0402327299825271;
                  } else {
                    result[0] += -0.030792413703587027;
                  }
                }
              }
            } else {
              result[0] += 0.036195589914049774;
            }
          } else {
            if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.527194023132325107) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.914472818374634233) ) ) {
                if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += 0.01421512827132879;
                } else {
                  if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += -0.016268544744821527;
                  } else {
                    result[0] += -0.055617107519720536;
                  }
                }
              } else {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.255632162094117099) ) ) {
                  result[0] += 0.0011313742372383016;
                } else {
                  if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.740319490432739702) ) ) {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.749434947967529741) ) ) {
                      result[0] += 0.000850887331861636;
                    } else {
                      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                        result[0] += -0.003480757465176019;
                      } else {
                        result[0] += -0.025309019191662608;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.321723937988282138) ) ) {
                          result[0] += -0.041900104395996;
                        } else {
                          result[0] += 0.01654159939315348;
                        }
                      } else {
                        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.868834793567657693) ) ) {
                          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                            result[0] += -0.0041742141935745365;
                          } else {
                            result[0] += 0.01524523937561035;
                          }
                        } else {
                          result[0] += -0.014795787947451122;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.102759599685669833) ) ) {
                        if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.617236852645874912) ) ) {
                            result[0] += -0.026099256565157655;
                          } else {
                            result[0] += -0.06251030859397858;
                          }
                        } else {
                          if ( UNLIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                            result[0] += 0.022630892568367998;
                          } else {
                            result[0] += -0.027131411169080234;
                          }
                        }
                      } else {
                        if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
                          result[0] += -0.013548912278936212;
                        } else {
                          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                            result[0] += 0.0001402020716468914;
                          } else {
                            result[0] += 0.0499054089875644;
                          }
                        }
                      }
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -5.8608075754435465e-05;
                } else {
                  result[0] += 0.06639638401212487;
                }
              } else {
                if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += 0.004996228642355272;
                  } else {
                    result[0] += -0.013053443917507483;
                  }
                } else {
                  result[0] += -0.01956348513536758;
                }
              }
            }
          }
        } else {
          result[0] += -0.009552979528231618;
        }
      } else {
        result[0] += -0.08131929679690227;
      }
    }
  }
  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.959391355514527255) ) ) {
        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
          if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.636499762535095659) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.880305767059327948) ) ) {
                  result[0] += -0.11219233396934909;
                } else {
                  result[0] += 0.06637356929819245;
                }
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.636499762535095659) ) ) {
                  result[0] += 0.09009047997507551;
                } else {
                  result[0] += 0.008256317633658646;
                }
              }
            } else {
              result[0] += -0.01036877062467874;
            }
          } else {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.433569431304932529) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.046861171722413886) ) ) {
                    result[0] += 0.011798494579901626;
                  } else {
                    if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += -0.10080174015504924;
                    } else {
                      result[0] += -0.02261760269949752;
                    }
                  }
                } else {
                  result[0] += -0.04293140585543598;
                }
              } else {
                result[0] += 0.005859935061966887;
              }
            } else {
              result[0] += 0.03446962767851094;
            }
          }
        } else {
          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
            if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += 0.002038708106084813;
              } else {
                if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.342454433441162998) ) ) {
                    result[0] += 0.06741040452662941;
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.843275547027588779) ) ) {
                      result[0] += -0.06913791978264046;
                    } else {
                      result[0] += -0.011853165536517322;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.497097015380861151) ) ) {
                    result[0] += 0.02322049475555502;
                  } else {
                    if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.221818685531617099) ) ) {
                      result[0] += 0.0031406691305296028;
                    } else {
                      if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                        result[0] += 0.15841951625257922;
                      } else {
                        result[0] += 0.005832783643031916;
                      }
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += 0.006098784428409601;
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.553712725639343706) ) ) {
                  result[0] += 0.08309091070695827;
                } else {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += -0.031414082520314635;
                    } else {
                      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += -0.12540024443558978;
                      } else {
                        result[0] += -0.009813853112369111;
                      }
                    }
                  } else {
                    result[0] += -0.013958084048507433;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.029068946838379794) ) ) {
                result[0] += 0.0010305893978751824;
              } else {
                if ( UNLIKELY(  (data[27].missing != -1) && (data[27].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                  result[0] += 0.07565982347620852;
                } else {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                    if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.329314231872559482) ) ) {
                        result[0] += -0.010134486221764587;
                      } else {
                        result[0] += -0.053864936012339895;
                      }
                    } else {
                      if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
                        result[0] += 0.041429118524984716;
                      } else {
                        if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
                          if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            result[0] += 0.007123277715680935;
                          } else {
                            result[0] += -0.016716212341747107;
                          }
                        } else {
                          result[0] += -0.045375627010147315;
                        }
                      }
                    }
                  } else {
                    result[0] += 0.004126922712899807;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5.007290840148926669) ) ) {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.012675821781158891) ) ) {
                    result[0] += -0.05414348106904329;
                  } else {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.778982400894165927) ) ) {
                      result[0] += -0.04585610843812378;
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.941167116165162021) ) ) {
                        result[0] += -0.06909060360946798;
                      } else {
                        result[0] += 0.03173986833380943;
                      }
                    }
                  }
                } else {
                  result[0] += 0.08725867043186085;
                }
              } else {
                if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.53326439857482999) ) ) {
                    result[0] += -0.08435444972611994;
                  } else {
                    result[0] += 0.16834418794448816;
                  }
                } else {
                  result[0] += -0.0848046911804756;
                }
              }
            }
          }
        }
      } else {
        result[0] += -0.013400011333984208;
      }
    } else {
      if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
        result[0] += -0.001290760615179098;
      } else {
        result[0] += 0.00211408228553919;
      }
    }
  } else {
    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.184114694595337802) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.129780292510988104) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.830334186553955966) ) ) {
            if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)3072.000000000000455) ) ) {
              result[0] += -0.023302496453908;
            } else {
              result[0] += 0.05812892597338607;
            }
          } else {
            result[0] += 0.0759164109204437;
          }
        } else {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.59476566314697443) ) ) {
            result[0] += 0.0021744033220740034;
          } else {
            result[0] += 0.08535712963055084;
          }
        }
      } else {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.718933820724488193) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.43267917633056818) ) ) {
            result[0] += 0.003251944758152365;
          } else {
            result[0] += 0.07105766683346978;
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.129040718078614169) ) ) {
            result[0] += -0.11181098603972935;
          } else {
            result[0] += -0.006405975606392435;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.87502956390381037) ) ) {
        result[0] += 0.001053366466137385;
      } else {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
          if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
            if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
              if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += 0.04462310498897595;
              } else {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.003200931425749485;
                } else {
                  if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.023252578453413886;
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.802696108818054643) ) ) {
                      result[0] += -0.007327914496076672;
                    } else {
                      result[0] += -0.11319376788735719;
                    }
                  }
                }
              }
            } else {
              result[0] += 0.007682361126754815;
            }
          } else {
            result[0] += -0.07015193732916769;
          }
        } else {
          result[0] += -0.028113486861234105;
        }
      }
    }
  }
  if ( UNLIKELY(  (data[31].missing != -1) && (data[31].fvalue <= (double)-1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.379217386245728427) ) ) {
      result[0] += 0.004775090836474877;
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.190353393554689276) ) ) {
        result[0] += -0.004730857968156892;
      } else {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.780892848968506748) ) ) {
          result[0] += 0.033989731407659686;
        } else {
          result[0] += 0.12748536634115482;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.617236852645874912) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.744487762451173651) ) ) {
          if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += 0.0008981692769955972;
          } else {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.23832273483276456) ) ) {
              result[0] += -0.0024107670312163293;
            } else {
              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.005727326512417885;
                } else {
                  result[0] += -0.043170166549508816;
                }
              } else {
                result[0] += -0.053441440838229805;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += -0.013322506850118227;
            } else {
              if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.088880300521851474) ) ) {
                  result[0] += -0.005912989591735645;
                } else {
                  result[0] += -0.0358860534137048;
                }
              } else {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.0007087990606059173;
                } else {
                  result[0] += 0.008953817989597245;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.938867926597595659) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.770631790161133257) ) ) {
                result[0] += -0.0026214376425258788;
              } else {
                result[0] += -0.038279667483716036;
              }
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.59476566314697443) ) ) {
                result[0] += 0.003506401920879846;
              } else {
                result[0] += 0.01671870613544875;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.474771499633789951) ) ) {
            if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += -0.031403173598726415;
            } else {
              result[0] += 0.014441046884440182;
            }
          } else {
            result[0] += -0.04273093044745891;
          }
        } else {
          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
            result[0] += -0.009772152978760905;
          } else {
            result[0] += 0.014374421224160445;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.744487762451173651) ) ) {
        result[0] += 0.00027409575887471396;
      } else {
        if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
          if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += -0.004990780645594478;
          } else {
            if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.532256603240968573) ) ) {
                  result[0] += 2.8613978995016793e-05;
                } else {
                  result[0] += 0.02109983086886008;
                }
              } else {
                if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                      result[0] += 0.0038168594967214102;
                    } else {
                      result[0] += -0.07416659119841304;
                    }
                  } else {
                    result[0] += -0.032613561596849715;
                  }
                } else {
                  if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.56941866874694913) ) ) {
                      result[0] += -0.032948643608567446;
                    } else {
                      result[0] += 0.005023857387195719;
                    }
                  } else {
                    result[0] += 0.06579223681886187;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)2.249904870986938921) ) ) {
                result[0] += 0.009370011682608126;
              } else {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += 0.11886622054153005;
                } else {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.13033212478260045;
                  } else {
                    result[0] += 0.06661341763049929;
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
                result[0] += 0.08111768898158532;
              } else {
                result[0] += -0.028588306191089898;
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
                result[0] += -0.06780399892992726;
              } else {
                result[0] += 0.03417737664583843;
              }
            }
          } else {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.962127923965454546) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.938867926597595659) ) ) {
                  if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                    if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.01913702311776908;
                    } else {
                      result[0] += 0.09037225090161936;
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.51517200469970881) ) ) {
                      result[0] += 0.034483458860744505;
                    } else {
                      result[0] += -0.05696112887179309;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.029068946838379794) ) ) {
                    if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)3.000000000000000444) ) ) {
                      result[0] += 0.0048038054678318206;
                    } else {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                        result[0] += -0.014127705936028447;
                      } else {
                        if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
                          result[0] += -0.046863739614621654;
                        } else {
                          result[0] += -0.021553788786061023;
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                        if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.765615224838257724) ) ) {
                          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.255632162094117099) ) ) {
                            result[0] += 0.003349903824288339;
                          } else {
                            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.780892848968506748) ) ) {
                              result[0] += -0.03176669081065014;
                            } else {
                              result[0] += -0.0015328481597706967;
                            }
                          }
                        } else {
                          result[0] += -0.02839835781674002;
                        }
                      } else {
                        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.07496595382690607) ) ) {
                          result[0] += 0.008546716882355488;
                        } else {
                          if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                            result[0] += -0.017955295210714033;
                          } else {
                            result[0] += 0.052963919216586254;
                          }
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                        result[0] += -0.00035382081553370304;
                      } else {
                        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                          result[0] += -0.0111669197611746;
                        } else {
                          if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                            result[0] += -0.01083241959234822;
                          } else {
                            result[0] += -0.04707864330994071;
                          }
                        }
                      }
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.241523027420044833) ) ) {
                  result[0] += -0.004284619645190272;
                } else {
                  result[0] += 0.026996198300450254;
                }
              }
            } else {
              result[0] += -0.05314156741188078;
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.617236852645874912) ) ) {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.744487762451173651) ) ) {
        if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
          result[0] += 0.003869006504087283;
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.594915628433228427) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.511434078216553178) ) ) {
              result[0] += 0.05956546022875521;
            } else {
              if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.347943067550660068) ) ) {
                    result[0] += 0.010046439129699929;
                  } else {
                    result[0] += -0.021124158659324013;
                  }
                } else {
                  result[0] += 0.022188471008190325;
                }
              } else {
                result[0] += -0.04643418367046979;
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.30853915214538663) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.924581527709961826) ) ) {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.384830474853516513) ) ) {
                  result[0] += -0.0063757234884112895;
                } else {
                  result[0] += -0.02802791567090637;
                }
              } else {
                result[0] += -0.00023032736710663516;
              }
            } else {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += 0.00017664669280525124;
                } else {
                  result[0] += 0.018162155711770638;
                }
              } else {
                result[0] += -0.0008471446870465347;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.39772605895996271) ) ) {
              result[0] += -0.008610245692039931;
            } else {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += 0.006634272655087305;
              } else {
                result[0] += 0.044973421244197835;
              }
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.28299736976623624) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
                result[0] += -0.031060561916237878;
              } else {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                  result[0] += -0.006517559458381653;
                } else {
                  result[0] += 0.012822608388941392;
                }
              }
            } else {
              result[0] += -0.013828915993168071;
            }
          }
        } else {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
            if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)3.000000000000000444) ) ) {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.00964093775377566;
              } else {
                result[0] += -0.0697275755414417;
              }
            } else {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.16594791412353693) ) ) {
                  if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.868834793567657693) ) ) {
                    result[0] += -0.07545621239573612;
                  } else {
                    result[0] += 0.003730631052878688;
                  }
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.381086945533752885) ) ) {
                    if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.11591521341502774;
                    } else {
                      result[0] += -0.00859119928375272;
                    }
                  } else {
                    result[0] += -0.023146325574106286;
                  }
                }
              } else {
                if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)14.19447278976440607) ) ) {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.242453336715698464) ) ) {
                      result[0] += -0.057882071660808304;
                    } else {
                      result[0] += -0.00467092256512922;
                    }
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.431901693344116655) ) ) {
                      result[0] += 0.030300216663493793;
                    } else {
                      result[0] += -0.031641880977117474;
                    }
                  }
                } else {
                  result[0] += 0.010504260630580822;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.262283086776734287) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.07465314865112482) ) ) {
                  if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                    if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.894094467163086826) ) ) {
                      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.531673669815064365) ) ) {
                        result[0] += -0.0016867324101989816;
                      } else {
                        if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += 0.06151611522106774;
                        } else {
                          result[0] += -0.031144930500683315;
                        }
                      }
                    } else {
                      result[0] += -0.05241977272622603;
                    }
                  } else {
                    result[0] += 0.03184970609541553;
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.938867926597595659) ) ) {
                    result[0] += -0.0068835062237728485;
                  } else {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                      if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.722943305969239169) ) ) {
                        if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.35061407089233576) ) ) {
                            result[0] += -0.024042932290209925;
                          } else {
                            result[0] += -0.13924394532346562;
                          }
                        } else {
                          result[0] += 0.0018061198569136036;
                        }
                      } else {
                        result[0] += 0.022677967444470204;
                      }
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.2121162414550799) ) ) {
                        result[0] += 0.012343542789756584;
                      } else {
                        if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += 0.029927639069076792;
                        } else {
                          result[0] += 0.07105383344181386;
                        }
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += 0.1602092878617434;
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.73821687698364435) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.467917680740357333) ) ) {
                      if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += 0.01211369024164323;
                      } else {
                        result[0] += -0.023881645599627205;
                      }
                    } else {
                      result[0] += -0.05134945548856934;
                    }
                  } else {
                    result[0] += -0.037952413726026785;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)1.497866153717041238) ) ) {
                result[0] += -0.07868288947090717;
              } else {
                if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.725620865821838823) ) ) {
                  result[0] += 0.003202258912261195;
                } else {
                  result[0] += 0.0083598447119108;
                }
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.474771499633789951) ) ) {
          if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)257681260544.0000305) ) ) {
            result[0] += -0.025544595554799255;
          } else {
            result[0] += 0.013859673518730232;
          }
        } else {
          result[0] += -0.030480388408713435;
        }
      } else {
        result[0] += 0.010007355607697594;
      }
    }
  } else {
    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
      result[0] += -0.010742875996414818;
    } else {
      if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.553712725639343706) ) ) {
        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.255632162094117099) ) ) {
          result[0] += 0.0005283027881249699;
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.36105370521545499) ) ) {
            result[0] += -0.00038303777683700306;
          } else {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
              result[0] += -0.005620303865905313;
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.102759599685669833) ) ) {
                result[0] += -0.047134948758059535;
              } else {
                result[0] += -0.0007980503284453952;
              }
            }
          }
        }
      } else {
        result[0] += -0.01944816304772244;
      }
    }
  }
  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.617236852645874912) ) ) {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.744487762451173651) ) ) {
        if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
          result[0] += 0.00101798401780155;
        } else {
          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.400584220886231357) ) ) {
                result[0] += 0.008147633283601614;
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.90474271774292081) ) ) {
                  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.002789756265206861;
                  } else {
                    if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.921060562133789951) ) ) {
                          result[0] += -0.02327216264425036;
                        } else {
                          result[0] += -0.05562135545487371;
                        }
                      } else {
                        result[0] += 0.018583336195284402;
                      }
                    } else {
                      result[0] += -0.0053222508029408915;
                    }
                  }
                } else {
                  result[0] += 0.0012801669272733069;
                }
              }
            } else {
              result[0] += -0.01701244699908345;
            }
          } else {
            if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.357691764831543413) ) ) {
              result[0] += -0.03232418394489094;
            } else {
              result[0] += 0.022474035746663415;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
              result[0] += -0.00047316051644077494;
            } else {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.267844915390015537) ) ) {
                result[0] += -0.005674507134403768;
              } else {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += -0.039317668661691785;
                } else {
                  result[0] += 0.021486905298342244;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.497097015380861151) ) ) {
                result[0] += 0.04924133155273485;
              } else {
                result[0] += 0.01117915197536309;
              }
            } else {
              result[0] += 0.0017075695230542893;
            }
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.938867926597595659) ) ) {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.770631790161133257) ) ) {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)2.012675821781158891) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.434600353240968573) ) ) {
                    result[0] += -0.16390269129192328;
                  } else {
                    result[0] += 0.008596150794189619;
                  }
                } else {
                  result[0] += -0.07339337549918638;
                }
              } else {
                result[0] += 0.011953298357509001;
              }
            } else {
              result[0] += -0.03389010547953316;
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.59476566314697443) ) ) {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.898905277252199042) ) ) {
                    result[0] += -0.03061070699084434;
                  } else {
                    result[0] += 0.006767718560497649;
                  }
                } else {
                  if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.031795776918934586;
                  } else {
                    result[0] += -0.00630957291665446;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.028175434467182594;
                    } else {
                      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                        result[0] += 0.02284859269879151;
                      } else {
                        if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.740319490432739702) ) ) {
                          result[0] += -0.030945423837503285;
                        } else {
                          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                            result[0] += -0.014928866728835245;
                          } else {
                            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.744781017303467685) ) ) {
                              result[0] += -0.033128412745742034;
                            } else {
                              if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                                result[0] += 0.00391555287682825;
                              } else {
                                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.357691764831543413) ) ) {
                                  result[0] += 0.050890429290262384;
                                } else {
                                  result[0] += -0.08616436526539031;
                                }
                              }
                            }
                          }
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += 0.03468019660462381;
                    } else {
                      result[0] += -0.02415911802135479;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
                    if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)6.25862455368042081) ) ) {
                      if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.855921268463135654) ) ) {
                        result[0] += 0.040125216857622915;
                      } else {
                        result[0] += 0.1361895982267883;
                      }
                    } else {
                      result[0] += -0.03010691194133074;
                    }
                  } else {
                    result[0] += 0.002780207728342598;
                  }
                }
              }
            } else {
              result[0] += 0.015429244948246863;
            }
          }
        }
      }
    } else {
      result[0] += 0.006464043141752725;
    }
  } else {
    if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.511434078216553178) ) ) {
        result[0] += 0.016380311721166493;
      } else {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
            result[0] += -0.01214379207146426;
          } else {
            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.510617971420288974) ) ) {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += -0.03703007426364209;
                } else {
                  if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.0033259164869283717;
                  } else {
                    result[0] += 0.07926593119969505;
                  }
                }
              } else {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
                    result[0] += 0.04374212259491726;
                  } else {
                    result[0] += -0.04318757477017636;
                  }
                } else {
                  result[0] += -0.01162767751741087;
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.248013019561768466) ) ) {
                result[0] += 0.08433080101745719;
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.53326439857482999) ) ) {
                  result[0] += -0.06361009945222389;
                } else {
                  result[0] += -0.02042032010853496;
                }
              }
            }
          }
        } else {
          result[0] += -0.02542892263846302;
        }
      }
    } else {
      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.35441589355468928) ) ) {
        result[0] += -7.5321306688496775e-06;
      } else {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
          result[0] += 0.0009745932094454156;
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.174569487571716753) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.94957673549652144) ) ) {
              result[0] += -0.03731013310093926;
            } else {
              result[0] += -0.013273501790821774;
            }
          } else {
            if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
              result[0] += -0.004301025385116393;
            } else {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)1.497866153717041238) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.770631790161133257) ) ) {
                  result[0] += 0.07609007019512647;
                } else {
                  result[0] += -0.01194068606827009;
                }
              } else {
                result[0] += -0.07634449349929567;
              }
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY(  (data[27].missing != -1) && (data[27].fvalue <= (double)-1.00000001800250948e-35) ) ) {
    result[0] += 0.009870526950019491;
  } else {
    if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
        if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.166635274887085849) ) ) {
          if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.863673448562622958) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.53326439857482999) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.69067406654357999) ) ) {
                result[0] += 0.012065131815607678;
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.219419956207276279) ) ) {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.531673669815064365) ) ) {
                    result[0] += -0.0669263386792467;
                  } else {
                    result[0] += 0.06927852527950039;
                  }
                } else {
                  result[0] += -0.10398537714310077;
                }
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.637949228286744052) ) ) {
                result[0] += 0.08087783226400462;
              } else {
                result[0] += 0.00946976656693837;
              }
            }
          } else {
            if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.01634240150451749) ) ) {
              result[0] += -0.014518405764411943;
            } else {
              result[0] += -0.10233938870363278;
            }
          }
        } else {
          result[0] += 0.10351833411455305;
        }
      } else {
        result[0] += -0.04641027751384245;
      }
    } else {
      if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.000000000000000444) ) ) {
        if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.532332420349121982) ) ) {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.909855604171753818) ) ) {
              if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)6.809154510498047763) ) ) {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.774904012680054599) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.262283086776734287) ) ) {
                      result[0] += 0.06162805775204871;
                    } else {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
                        result[0] += -0.08347785609641672;
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.247576236724854404) ) ) {
                          result[0] += 0.08216581186049768;
                        } else {
                          result[0] += 0.008197543610215834;
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                      result[0] += 0.120426948113038;
                    } else {
                      result[0] += 0.019419847127964036;
                    }
                  }
                } else {
                  result[0] += -0.06534734966481263;
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.088880300521851474) ) ) {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.881510615348816362) ) ) {
                    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.733271598815919745) ) ) {
                        result[0] += 0.09401345384514267;
                      } else {
                        result[0] += -0.028282952960314536;
                      }
                    } else {
                      if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.31402075290679976) ) ) {
                        result[0] += -0.12774089584093803;
                      } else {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.397998809814454013) ) ) {
                          result[0] += -0.061845693230862135;
                        } else {
                          result[0] += 0.061603034527682426;
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.35441589355468928) ) ) {
                        result[0] += -0.1859416923428287;
                      } else {
                        result[0] += -0.061697987753128715;
                      }
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.510617971420288974) ) ) {
                        result[0] += -0.11347165145592684;
                      } else {
                        result[0] += 0.0591497357751023;
                      }
                    }
                  }
                } else {
                  result[0] += 0.042638630251522494;
                }
              }
            } else {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.74845767021179288) ) ) {
                  result[0] += -0.02614669331324243;
                } else {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.625819444656372958) ) ) {
                    if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.58961367607116788) ) ) {
                      result[0] += 0.10516963090414477;
                    } else {
                      if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.735185861587525302) ) ) {
                        result[0] += -0.046188074245250475;
                      } else {
                        if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += 0.039368478434032385;
                        } else {
                          result[0] += 0.1617665660635612;
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.569433569908142534) ) ) {
                      result[0] += 0.09131325817250596;
                    } else {
                      result[0] += -0.12145350042034223;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.184114694595337802) ) ) {
                  result[0] += 0.07529851091788352;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.43749904632568537) ) ) {
                    result[0] += 0.08067127745607751;
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.909254074096680576) ) ) {
                      result[0] += -0.05181720068638923;
                    } else {
                      result[0] += 0.01583294049847262;
                    }
                  }
                }
              }
            }
          } else {
            result[0] += -0.0572424004420546;
          }
        } else {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
            if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.848108768463135654) ) ) {
              if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.010941914244997039;
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.241523027420044833) ) ) {
                  result[0] += 0.0031348727613808866;
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.855006217956543857) ) ) {
                    result[0] += -0.1403583567376353;
                  } else {
                    result[0] += -0.012513184540803147;
                  }
                }
              }
            } else {
              result[0] += 0.04591953112345364;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.589234352111818183) ) ) {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.780892848968506748) ) ) {
                result[0] += 0.031079494095450872;
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.029068946838379794) ) ) {
                  result[0] += -0.10913349366885892;
                } else {
                  result[0] += -0.0129166465257726;
                }
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.531673669815064365) ) ) {
                result[0] += 0.009215770320471887;
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.219419956207276279) ) ) {
                  result[0] += -0.09556878378844874;
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.397998809814454013) ) ) {
                    result[0] += 0.07033761249202104;
                  } else {
                    result[0] += -0.058142169665876375;
                  }
                }
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += -0.000649108390267066;
        } else {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
            result[0] += 0.0010296486726047815;
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.510617971420288974) ) ) {
              result[0] += 0.021094059646821595;
            } else {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.050296032217852664;
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.54220247268676935) ) ) {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.835998296737671787) ) ) {
                    result[0] += -0.0433329933395856;
                  } else {
                    if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.582417964935304511) ) ) {
                      if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.449861526489258257) ) ) {
                        result[0] += -0.011184661363218659;
                      } else {
                        result[0] += 0.018179123030621826;
                      }
                    } else {
                      if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                        result[0] += -0.047733142850003366;
                      } else {
                        result[0] += -0.27387099083576416;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.993164777755738193) ) ) {
                    result[0] += 0.008221550945446665;
                  } else {
                    result[0] += 0.0890687236467669;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY(  (data[41].missing != -1) && (data[41].fvalue <= (double)-1.00000001800250948e-35) ) ) {
    result[0] += 0.0024892967041747344;
  } else {
    if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)3.000000000000000444) ) ) {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.537586927413940874) ) ) {
          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
            result[0] += -0.045682845958767836;
          } else {
            if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)7.508512496948243076) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.329314231872559482) ) ) {
                result[0] += -0.0029421529445150356;
              } else {
                result[0] += -0.019156482883699505;
              }
            } else {
              result[0] += 0.08486604644163626;
            }
          }
        } else {
          result[0] += -0.03172314595415097;
        }
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
          if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
            if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.636499762535095659) ) ) {
                  if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                    result[0] += -0.13282316661735086;
                  } else {
                    result[0] += 0.03342871783269673;
                  }
                } else {
                  if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.012675821781158891) ) ) {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.216319084167481357) ) ) {
                      result[0] += -0.13063955721065634;
                    } else {
                      result[0] += 0.02565468176053229;
                    }
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.53326439857482999) ) ) {
                      result[0] += 0.01242063007069821;
                    } else {
                      result[0] += -0.002940521965884949;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.012675821781158891) ) ) {
                    if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                      result[0] += 0.03262445582071115;
                    } else {
                      result[0] += -0.015288120531014816;
                    }
                  } else {
                    if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)6.000000000000000888) ) ) {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.51517200469970881) ) ) {
                        result[0] += -0.0446922497841169;
                      } else {
                        result[0] += 0.01030592532910889;
                      }
                    } else {
                      result[0] += -0.011208186630175344;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.534971714019776279) ) ) {
                    if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                      result[0] += 0.005974864313552014;
                    } else {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.827801465988160068) ) ) {
                        result[0] += -0.0025294519230138306;
                      } else {
                        result[0] += -0.03993167951291987;
                      }
                    }
                  } else {
                    result[0] += 0.014934726622244621;
                  }
                }
              }
            } else {
              result[0] += 0.052955050086574786;
            }
          } else {
            result[0] += -0.00036584764882605694;
          }
        } else {
          result[0] += 0.0006285184466956104;
        }
      }
    } else {
      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.379217386245728427) ) ) {
        result[0] += -0.0003872532989375117;
      } else {
        if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.740319490432739702) ) ) {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.436733961105347568) ) ) {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += 0.025433347687842042;
                } else {
                  result[0] += -0.008605746150138725;
                }
              } else {
                result[0] += 0.016447806749025372;
              }
            } else {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += 0.026654754200110178;
                } else {
                  result[0] += -0.006524559896197945;
                }
              } else {
                if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += -0.004475463600088067;
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.637949228286744052) ) ) {
                        result[0] += 0.01226422053109219;
                      } else {
                        result[0] += 0.059586158005053215;
                      }
                    }
                  } else {
                    result[0] += 0.008298844811337523;
                  }
                } else {
                  result[0] += -6.417881423884964e-05;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.497097015380861151) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)8.285748958587648261) ) ) {
                if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                  if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.843275547027588779) ) ) {
                      result[0] += 0.022545744914294816;
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.24173307418823331) ) ) {
                        result[0] += 0.046345416649159685;
                      } else {
                        result[0] += -0.10074395595836523;
                      }
                    }
                  } else {
                    result[0] += 0.016798863543560957;
                  }
                } else {
                  result[0] += 0.0014712978021652488;
                }
              } else {
                result[0] += -0.05536287345845925;
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.241523027420044833) ) ) {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                  result[0] += -0.009801517010262877;
                } else {
                  result[0] += 0.034260090938909235;
                }
              } else {
                if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += 0.028416742562532977;
                  } else {
                    if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += -0.023210264129476424;
                    } else {
                      result[0] += -0.07253677356120079;
                    }
                  }
                } else {
                  result[0] += -0.013854432318815033;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.1097903251647967) ) ) {
            if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
              if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
                  result[0] += 0.0019246584699963024;
                } else {
                  if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                    if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += -0.06775898579447029;
                    } else {
                      result[0] += 0.011285320713435203;
                    }
                  } else {
                    result[0] += -0.03942029433352964;
                  }
                }
              } else {
                if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.09490379237926307;
                } else {
                  result[0] += 0.01785455388189654;
                }
              }
            } else {
              if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.321723937988282138) ) ) {
                  result[0] += 0.021485166976272233;
                } else {
                  result[0] += 0.0640108597049978;
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.431901693344116655) ) ) {
                  result[0] += 0.11983804902383542;
                } else {
                  result[0] += 0.0030730429613204896;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.637949228286744052) ) ) {
              result[0] += 0.0052920949043654825;
            } else {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.024760979723110896;
                } else {
                  result[0] += -0.008043714244809054;
                }
              } else {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.959391355514527255) ) ) {
                    result[0] += 0.044091825369060055;
                  } else {
                    result[0] += -0.04085561156534581;
                  }
                } else {
                  result[0] += 0.03946253230162508;
                }
              }
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY(  (data[40].missing != -1) && (data[40].fvalue <= (double)-1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5.453179836273194248) ) ) {
      if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.384830474853516513) ) ) {
          if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                      result[0] += -0.07371858348849149;
                    } else {
                      result[0] += 0.0029480629354403823;
                    }
                  } else {
                    result[0] += 0.014637789328360013;
                  }
                } else {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
                    result[0] += 0.004349081874911673;
                  } else {
                    result[0] += -0.019860345650562442;
                  }
                }
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.855006217956543857) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.422362327575684482) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.733271598815919745) ) ) {
                      if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.673553824424744096) ) ) {
                        result[0] += -0.023198258504886535;
                      } else {
                        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
                          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
                            result[0] += 0.13008348271924247;
                          } else {
                            result[0] += -0.04667375455139423;
                          }
                        } else {
                          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.070700883865357333) ) ) {
                            if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.142630577087403232) ) ) {
                                result[0] += -0.04997371464634328;
                              } else {
                                result[0] += 0.03399519436735065;
                              }
                            } else {
                              result[0] += -0.043145828187849615;
                            }
                          } else {
                            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                              result[0] += 0.10879315594890122;
                            } else {
                              result[0] += -0.005360752348358652;
                            }
                          }
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
                        result[0] += -0.03101010429072848;
                      } else {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.802696108818054643) ) ) {
                          result[0] += -0.01408850800397967;
                        } else {
                          if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)6.000000000000000888) ) ) {
                            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.637949228286744052) ) ) {
                              result[0] += -0.002410823997678815;
                            } else {
                              result[0] += 0.056288763244056156;
                            }
                          } else {
                            if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.349750161170959917) ) ) {
                              result[0] += 0.03750619443449782;
                            } else {
                              result[0] += 0.09161569677261107;
                            }
                          }
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += 0.01497932903959372;
                    } else {
                      result[0] += 0.08377587427789032;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.623839378356934482) ) ) {
                      result[0] += 0.10044185161037045;
                    } else {
                      result[0] += 0.016576990818065775;
                    }
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.843275547027588779) ) ) {
                      result[0] += -0.0713786920460566;
                    } else {
                      result[0] += 0.08173793309486097;
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.909102678298951083) ) ) {
                  result[0] += 0.07191659355402784;
                } else {
                  result[0] += -0.06802082216650596;
                }
              } else {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.855921268463135654) ) ) {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.715336322784424716) ) ) {
                        result[0] += 0.0648650526290994;
                      } else {
                        result[0] += -0.035584805208497355;
                      }
                    } else {
                      result[0] += -0.09085586419998738;
                    }
                  } else {
                    result[0] += -0.09942929038757693;
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.53326439857482999) ) ) {
                    result[0] += 0.07121425062686502;
                  } else {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.881510615348816362) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.216319084167481357) ) ) {
                        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
                          result[0] += -0.021467187081214444;
                        } else {
                          result[0] += -0.1223194321385933;
                        }
                      } else {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.136462926864624912) ) ) {
                          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.241523027420044833) ) ) {
                            result[0] += 0.06380311249419086;
                          } else {
                            result[0] += -0.024517510336682233;
                          }
                        } else {
                          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.289595603942871982) ) ) {
                            result[0] += -0.09294536771068468;
                          } else {
                            result[0] += 0.020604268281566716;
                          }
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.733271598815919745) ) ) {
                          result[0] += 0.07696739768012993;
                        } else {
                          result[0] += -0.05427180708316707;
                        }
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.835998296737671787) ) ) {
                          result[0] += -0.06086428563954899;
                        } else {
                          result[0] += 0.10986074096117449;
                        }
                      }
                    }
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.046861171722413886) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.342454433441162998) ) ) {
                result[0] += 0.04590276592267856;
              } else {
                result[0] += 0.015928437907337088;
              }
            } else {
              if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += 0.005130222882192169;
                } else {
                  result[0] += -0.011228550003221467;
                }
              } else {
                result[0] += 0.015103099344549656;
              }
            }
          }
        } else {
          result[0] += -0.004055599833236511;
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.982408046722412998) ) ) {
          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.53326439857482999) ) ) {
              result[0] += -0.13648118213732027;
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += -0.0528206428154288;
              } else {
                if ( UNLIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.11353272647123805;
                } else {
                  result[0] += -0.0007740994047052349;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
              result[0] += -0.11139661351226125;
            } else {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
                result[0] += 0.020011911300464008;
              } else {
                result[0] += -0.08766772254159978;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.855921268463135654) ) ) {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.770631790161133257) ) ) {
              result[0] += -0.05160743202288496;
            } else {
              result[0] += 0.0004914634159170018;
            }
          } else {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.778982400894165927) ) ) {
                result[0] += -0.1738023893986652;
              } else {
                result[0] += 0.01085193060577569;
              }
            } else {
              result[0] += 0.031940538111110896;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5.486867427825928623) ) ) {
        result[0] += 0.1157040975861987;
      } else {
        result[0] += 0.01743794326128189;
      }
    }
  } else {
    result[0] += -0.0001994561487424285;
  }
  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
    if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
        if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)7.508512496948243076) ) ) {
          result[0] += 0.004458387618292953;
        } else {
          if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
            result[0] += 0.014382205795366705;
          } else {
            result[0] += 0.18833685371133296;
          }
        }
      } else {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.41263532638549982) ) ) {
          if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
            result[0] += -0.0005966271152342464;
          } else {
            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.249904870986938921) ) ) {
              result[0] += -0.080660796231636;
            } else {
              result[0] += -0.010571015539095592;
            }
          }
        } else {
          result[0] += -0.08356521609847006;
        }
      }
    } else {
      if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
        if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
          if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)2.071567356586456743) ) ) {
            result[0] += -0.0014027083574675678;
          } else {
            result[0] += -0.06893881862245275;
          }
        } else {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.088880300521851474) ) ) {
            if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.868834793567657693) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += 0.17184257619802557;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.094205617904663974) ) ) {
                  result[0] += 0.027937824451458232;
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.2531323432922381) ) ) {
                    result[0] += -0.008149017469081808;
                  } else {
                    result[0] += -0.0713149109684904;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                result[0] += 0.06182358795241194;
              } else {
                result[0] += -0.06897027085180259;
              }
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.53326439857482999) ) ) {
              result[0] += 0.010762980174683936;
            } else {
              result[0] += -0.05950895216837097;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
              result[0] += 0.0028412070419561054;
            } else {
              result[0] += -0.0422613429254931;
            }
          } else {
            if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += 0.00951601338204884;
            } else {
              result[0] += -0.016859023006805217;
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.368446350097658026) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.055496215820313388) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.511434078216553178) ) ) {
                result[0] += 0.05199607293787766;
              } else {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.007531327071516997;
                } else {
                  result[0] += 0.014583983846631319;
                }
              }
            } else {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.288254261016846591) ) ) {
                  if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
                    result[0] += -0.009875508375003339;
                  } else {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.329314231872559482) ) ) {
                      if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += 0.0011267488871084166;
                      } else {
                        result[0] += 0.09326170611232931;
                      }
                    } else {
                      result[0] += 0.0034730348147903115;
                    }
                  }
                } else {
                  result[0] += 0.0036998622139933443;
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.594915628433228427) ) ) {
                  result[0] += 0.07177200515199607;
                } else {
                  if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.856657028198243964) ) ) {
                    result[0] += -0.016695933409577528;
                  } else {
                    result[0] += -0.1623827801182353;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.2531323432922381) ) ) {
                  result[0] += -0.009346279912489755;
                } else {
                  result[0] += 0.02476487599674147;
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.154959201812744585) ) ) {
                  result[0] += 0.02946316733710351;
                } else {
                  result[0] += -0.040483015793778755;
                }
              }
            } else {
              if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.835998296737671787) ) ) {
                  result[0] += 0.02115330728936265;
                } else {
                  if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.005609880436152352;
                  } else {
                    result[0] += -0.04221316475427178;
                  }
                }
              } else {
                result[0] += 0.004738514517439359;
              }
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.129040718078614169) ) ) {
          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
            result[0] += -0.004965795062113429;
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.531673669815064365) ) ) {
              result[0] += -0.022431027990727173;
            } else {
              result[0] += 0.0405552205414254;
            }
          }
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.174569487571716753) ) ) {
            if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)3.000000000000000444) ) ) {
              result[0] += 0.024393340653894303;
            } else {
              result[0] += -0.013987753684107502;
            }
          } else {
            result[0] += 0.012576449749699415;
          }
        }
      } else {
        result[0] += -0.04084076103061523;
      }
    } else {
      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.44381141662597834) ) ) {
        result[0] += 3.456571716215609e-05;
      } else {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.803987503051758701) ) ) {
              result[0] += -0.02510605927493455;
            } else {
              result[0] += -0.00568008941734026;
            }
          } else {
            if ( UNLIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.53326439857482999) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.938867926597595659) ) ) {
                    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += -0.10817519209468268;
                    } else {
                      result[0] += 0.05392105039579236;
                    }
                  } else {
                    result[0] += -0.006883838339977687;
                  }
                } else {
                  result[0] += -0.06667866025208695;
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.431901693344116655) ) ) {
                  result[0] += -0.03633961576297513;
                } else {
                  result[0] += 0.01921293454292955;
                }
              }
            } else {
              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.219419956207276279) ) ) {
                  result[0] += -0.004728920214055044;
                } else {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += -0.010846102797421486;
                  } else {
                    result[0] += 0.046890486119612565;
                  }
                }
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.23832273483276456) ) ) {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += 0.032333379082755194;
                  } else {
                    result[0] += -0.03192365873398503;
                  }
                } else {
                  result[0] += 0.017714630555106445;
                }
              }
            }
          }
        } else {
          result[0] += -0.026169511492910316;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
        result[0] += 0.004915145413851681;
      } else {
        if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
          result[0] += -0.11200091905783255;
        } else {
          result[0] += 0.034538090574968865;
        }
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.94957673549652144) ) ) {
        if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
          result[0] += -0.003774711445942193;
        } else {
          result[0] += -0.02301529569048646;
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.921924352645874468) ) ) {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
            result[0] += -0.0063224446528807494;
          } else {
            result[0] += -0.03913983049012767;
          }
        } else {
          if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
              if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.774904012680054599) ) ) {
                result[0] += -0.011804179445522704;
              } else {
                result[0] += -0.12964891798935524;
              }
            } else {
              if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += 0.0027871387933148173;
              } else {
                if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.79134798049926935) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.940167903900147373) ) ) {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.384830474853516513) ) ) {
                        result[0] += -0.07543287929137554;
                      } else {
                        result[0] += 0.0486420712656499;
                      }
                    } else {
                      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.695749998092652255) ) ) {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.030070781707765448) ) ) {
                            result[0] += -0.008260531917469728;
                          } else {
                            if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                              if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                                result[0] += -0.02037222765817496;
                              } else {
                                result[0] += 0.05942210612015745;
                              }
                            } else {
                              result[0] += 0.06710663843413121;
                            }
                          }
                        } else {
                          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.967588424682618964) ) ) {
                            result[0] += 0.06986815641641946;
                          } else {
                            result[0] += -0.010740571509618477;
                          }
                        }
                      } else {
                        result[0] += 0.0627214677328543;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
                      result[0] += 0.08357980415756752;
                    } else {
                      result[0] += -0.04514441309035965;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                    result[0] += -0.011063170922438864;
                  } else {
                    if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.511434078216553178) ) ) {
                      result[0] += 0.003791237257887023;
                    } else {
                      result[0] += 0.035047765562049076;
                    }
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.102759599685669833) ) ) {
                if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.06080361900153794;
                } else {
                  if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += 0.004538674831090492;
                      } else {
                        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.088880300521851474) ) ) {
                          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.91907978057861506) ) ) {
                            result[0] += 0.009532190205763397;
                          } else {
                            result[0] += -0.019643757633019218;
                          }
                        } else {
                          result[0] += -0.009128797194003302;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.35441589355468928) ) ) {
                          if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                            result[0] += -0.0008817009082265073;
                          } else {
                            result[0] += -0.035620364003447945;
                          }
                        } else {
                          result[0] += 0.001207760395531026;
                        }
                      } else {
                        result[0] += 0.034817508239002225;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += -0.020707231073417973;
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.918693304061890537) ) ) {
                        result[0] += -0.03162401853688739;
                      } else {
                        result[0] += 0.003723752914929365;
                      }
                    }
                  }
                }
              } else {
                result[0] += -0.03139527621707813;
              }
            } else {
              if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)3.449861526489258257) ) ) {
                  result[0] += 0.07437062919717967;
                } else {
                  result[0] += -0.01404370115107024;
                }
              } else {
                if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.363078355789185458) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.617236852645874912) ) ) {
                    result[0] += 0.0003778573492108559;
                  } else {
                    if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                        result[0] += 0.1434322027339248;
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.088880300521851474) ) ) {
                          result[0] += 0.008587037220273415;
                        } else {
                          result[0] += -0.035318238974889636;
                        }
                      }
                    } else {
                      result[0] += 0.01419217955488631;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                        result[0] += 0.10343234115261346;
                      } else {
                        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                            result[0] += 0.013832235494804208;
                          } else {
                            result[0] += -0.0944890270779719;
                          }
                        } else {
                          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                            result[0] += -0.0029955660274235754;
                          } else {
                            result[0] += 0.11911168190567117;
                          }
                        }
                      }
                    } else {
                      result[0] += -0.07212383156839239;
                    }
                  } else {
                    if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                        result[0] += 0.006736195472229742;
                      } else {
                        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
                          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                            result[0] += 0.09870210106666988;
                          } else {
                            result[0] += -0.0722745578721332;
                          }
                        } else {
                          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.22084951400757014) ) ) {
                            result[0] += 0.07577403289110016;
                          } else {
                            result[0] += 0.02012370751190456;
                          }
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)12.00000000000000178) ) ) {
                        if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.48298668861389249) ) ) {
                          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                            result[0] += -0.013794458519936718;
                          } else {
                            result[0] += 0.010471305143181875;
                          }
                        } else {
                          result[0] += 0.01821706999851455;
                        }
                      } else {
                        result[0] += 0.035418672968343744;
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
    if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
        result[0] += 5.08964666099942e-05;
      } else {
        result[0] += -0.014384898131377067;
      }
    } else {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
        result[0] += -0.00514035712217516;
      } else {
        result[0] += 1.6315639144792995e-05;
      }
    }
  }
  if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)11.08291149139404475) ) ) {
      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.182065486907959873) ) ) {
        if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.909102678298951083) ) ) {
            if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.126885652542115146) ) ) {
                result[0] += 0.025693780189739238;
              } else {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.24173307418823331) ) ) {
                    result[0] += 0.030191485894301248;
                  } else {
                    result[0] += -0.01966836281443629;
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.867504835128785068) ) ) {
                    if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += 0.028061969989372854;
                    } else {
                      result[0] += -0.09368635562468662;
                    }
                  } else {
                    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += -0.0016085335310880038;
                    } else {
                      result[0] += 0.07250317330880038;
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += 0.0066154500704961846;
              } else {
                result[0] += -0.013080170804033437;
              }
            }
          } else {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.318498134613038886) ) ) {
                    result[0] += 0.049553987613013356;
                  } else {
                    result[0] += -0.0056911358829525045;
                  }
                } else {
                  if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)6.360871315002442294) ) ) {
                      result[0] += 0.022113004191964318;
                    } else {
                      result[0] += 0.10503804800147375;
                    }
                  } else {
                    result[0] += 0.13183261078764116;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.431901693344116655) ) ) {
                  result[0] += -0.06931558037808726;
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.835998296737671787) ) ) {
                    if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.511434078216553178) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.288152217864991123) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.041921615600587714) ) ) {
                          result[0] += 0.12027079718550655;
                        } else {
                          result[0] += 0.030096804286943824;
                        }
                      } else {
                        result[0] += -0.009976418696808131;
                      }
                    } else {
                      result[0] += -0.01623797562008079;
                    }
                  } else {
                    result[0] += -0.020061056747789918;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.82428741455078303) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.025192260742188388) ) ) {
                    result[0] += -0.1091705932435983;
                  } else {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.153024196624756748) ) ) {
                      if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2415.000000000000455) ) ) {
                        result[0] += 0.07263842781426975;
                      } else {
                        result[0] += -0.02444113735786978;
                      }
                    } else {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
                        result[0] += -0.14983393709639567;
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.464211463928224433) ) ) {
                          result[0] += 0.027832140745539864;
                        } else {
                          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.665476083755494052) ) ) {
                            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.48738741874694913) ) ) {
                              result[0] += -0.03204716485552233;
                            } else {
                              result[0] += 0.0907201760047178;
                            }
                          } else {
                            result[0] += -0.07457440622952781;
                          }
                        }
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.993164777755738193) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.51517200469970881) ) ) {
                      result[0] += 0.09582399165249987;
                    } else {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.95386886596679865) ) ) {
                        result[0] += -0.050501068089182845;
                      } else {
                        if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                          result[0] += 0.10924664151872193;
                        } else {
                          result[0] += -0.020183150927202423;
                        }
                      }
                    }
                  } else {
                    result[0] += 0.10155845918234441;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)8816427008.000001907) ) ) {
                  result[0] += 0.12669686302074887;
                } else {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.025192260742188388) ) ) {
                      result[0] += -0.011284561645522086;
                    } else {
                      if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                        result[0] += 0.021834812411181737;
                      } else {
                        result[0] += 0.06614945783105013;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.262283086776734287) ) ) {
                      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                        result[0] += 0.014964827364490091;
                      } else {
                        result[0] += -0.03936324960728024;
                      }
                    } else {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.01293420791626154) ) ) {
                        result[0] += 0.006766417231028943;
                      } else {
                        if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                          result[0] += 0.08811968660930741;
                        } else {
                          result[0] += 0.009818519983838642;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.36986422538757413) ) ) {
            result[0] += -0.14224921711138447;
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.847591876983644354) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.597323656082154208) ) ) {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.055311203002930576) ) ) {
                    result[0] += -0.06990791662192862;
                  } else {
                    result[0] += -0.004054364399289591;
                  }
                } else {
                  result[0] += -0.05261788770411732;
                }
              } else {
                result[0] += 0.0633859489187497;
              }
            } else {
              result[0] += 0.00865760103517612;
            }
          }
        }
      } else {
        if ( UNLIKELY(  (data[27].missing != -1) && (data[27].fvalue <= (double)-1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
            result[0] += -0.10590937138910879;
          } else {
            result[0] += 0.04150693949256384;
          }
        } else {
          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
            if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
              result[0] += -0.04346959913486456;
            } else {
              result[0] += 0.02230038799111838;
            }
          } else {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
              if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += -0.00023733078190716793;
              } else {
                if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.138333082199097124) ) ) {
                  result[0] += -0.060668494204695805;
                } else {
                  result[0] += -0.0161629388354179;
                }
              }
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += -0.07436702690528206;
                  } else {
                    result[0] += 0.024611453683184546;
                  }
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.623839378356934482) ) ) {
                    result[0] += -0.08959497125366073;
                  } else {
                    result[0] += -0.0008264975840146273;
                  }
                }
              } else {
                result[0] += 0.004605176198577831;
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.76779222488403498) ) ) {
        result[0] += 0.17121289148764018;
      } else {
        result[0] += 0.03306721841654127;
      }
    }
  } else {
    result[0] += -0.0002479500161063829;
  }
  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
      result[0] += 0.00430252104735725;
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.94957673549652144) ) ) {
        if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.0034844513131866307;
            } else {
              result[0] += -0.03348682303238095;
            }
          } else {
            if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += 0.004378564245037454;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.289521217346193183) ) ) {
                result[0] += -0.051049328163393415;
              } else {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += 0.009452471772609203;
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)16.83253097534180043) ) ) {
                      if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)6.743430852890015537) ) ) {
                        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)6.809154510498047763) ) ) {
                          result[0] += -0.03325711627687751;
                        } else {
                          result[0] += -0.14848511603451292;
                        }
                      } else {
                        result[0] += 0.05828730500790918;
                      }
                    } else {
                      result[0] += 0.043965238091844605;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += 0.07886942983262643;
                  } else {
                    result[0] += -0.029801313763785576;
                  }
                }
              }
            }
          }
        } else {
          result[0] += -0.021809799660563962;
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.921924352645874468) ) ) {
          result[0] += -0.010365995159990695;
        } else {
          result[0] += 0.0008377593253847864;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.750972747802735263) ) ) {
          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
            result[0] += -0.0037467332560741557;
          } else {
            result[0] += 0.02975395858944704;
          }
        } else {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
            if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.700598716735840066) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.426736354827881748) ) ) {
                if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)7.211187601089478427) ) ) {
                  if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += -0.01710477287440087;
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
                      result[0] += -0.019562918403360482;
                    } else {
                      if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                          result[0] += 0.0971122421219825;
                        } else {
                          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                            result[0] += 0.029752892375568735;
                          } else {
                            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                              result[0] += -0.10442089578418577;
                            } else {
                              result[0] += 0.01677834784353637;
                            }
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)4.579839229583741123) ) ) {
                          result[0] += 0.11411388950044675;
                        } else {
                          if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.993822574615479404) ) ) {
                            if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.980170249938965732) ) ) {
                              result[0] += -0.004459722092237162;
                            } else {
                              result[0] += 0.17393400374139797;
                            }
                          } else {
                            if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.650573849678039995) ) ) {
                              result[0] += 0.013905124784917837;
                            } else {
                              result[0] += -0.10858992351596511;
                            }
                          }
                        }
                      }
                    }
                  }
                } else {
                  result[0] += -0.08065599256007704;
                }
              } else {
                result[0] += 0.03698682651849384;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.055311203002930576) ) ) {
                result[0] += -0.05099975623348884;
              } else {
                result[0] += 0.04940074874421958;
              }
            }
          } else {
            result[0] += -0.01669920514129124;
          }
        }
      } else {
        result[0] += -0.05749927119252438;
      }
    } else {
      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.44381141662597834) ) ) {
        if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)6.000000000000000888) ) ) {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += 0.07586562005378414;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.556798219680787021) ) ) {
                result[0] += -0.017276187387256028;
              } else {
                result[0] += 0.0402529330608043;
              }
            }
          } else {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
              if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)10.89387273788452326) ) ) {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.872538805007935458) ) ) {
                  result[0] += 0.0014295567733151019;
                } else {
                  result[0] += -0.06542165686225072;
                }
              } else {
                result[0] += 0.1203059942105333;
              }
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.142630577087403232) ) ) {
                result[0] += -0.061214154294210024;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.921924352645874468) ) ) {
                  result[0] += -0.06419034669439068;
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.53326439857482999) ) ) {
                    result[0] += 0.017458860996384207;
                  } else {
                    result[0] += 0.08305197651283311;
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += 0.002752296258905515;
            } else {
              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.020127415657043901) ) ) {
                    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                      result[0] += -0.015885369161683864;
                    } else {
                      result[0] += 0.014503258232657485;
                    }
                  } else {
                    result[0] += 0.017374889425080635;
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.381086945533752885) ) ) {
                    result[0] += 0.027195924669313077;
                  } else {
                    result[0] += -0.033004435283993674;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.431901693344116655) ) ) {
                  result[0] += 0.051626077914861804;
                } else {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += -0.007374381585062463;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.654679536819458896) ) ) {
                      result[0] += -0.013517501557798094;
                    } else {
                      result[0] += -0.04485448874651703;
                    }
                  }
                }
              }
            }
          } else {
            result[0] += 0.000408965632488978;
          }
        }
      } else {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
          result[0] += 0.00044463075129031826;
        } else {
          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)1.242453336715698464) ) ) {
            if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.700598716735840066) ) ) {
              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                  result[0] += -0.04093604734864667;
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.602003335952759233) ) ) {
                    result[0] += -0.050434587642697204;
                  } else {
                    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                      result[0] += -0.05877114642402657;
                    } else {
                      result[0] += -0.00034915439371752333;
                    }
                  }
                }
              } else {
                result[0] += -0.0247680328558293;
              }
            } else {
              result[0] += 0.0044553290065609714;
            }
          } else {
            result[0] += -0.05739538645872736;
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.12109565734863459) ) ) {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
        result[0] += 0.007332765129177363;
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.288254261016846591) ) ) {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)8.310138225555421698) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.531673669815064365) ) ) {
              result[0] += -0.008141211540170891;
            } else {
              if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += 0.011672884717786438;
              } else {
                result[0] += -0.0047732853779629495;
              }
            }
          } else {
            result[0] += 0.10549500089749153;
          }
        } else {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.921060562133789951) ) ) {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.384830474853516513) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.041387319564820224) ) ) {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.189540147781372958) ) ) {
                      result[0] += -0.018113471175149456;
                    } else {
                      result[0] += 0.034908098740038765;
                    }
                  } else {
                    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
                        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                          result[0] += 0.018671358873787076;
                        } else {
                          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                            result[0] += -0.04331571143792992;
                          } else {
                            result[0] += -0.1434169061988149;
                          }
                        }
                      } else {
                        result[0] += -0.015203362815675353;
                      }
                    } else {
                      if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                        result[0] += -0.026267402157763534;
                      } else {
                        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                          if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            result[0] += 0.007777626731646592;
                          } else {
                            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                              result[0] += 0.09236401151870408;
                            } else {
                              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
                                result[0] += 0.08050848755274354;
                              } else {
                                result[0] += -0.006484127873042348;
                              }
                            }
                          }
                        } else {
                          result[0] += -0.01603897463547233;
                        }
                      }
                    }
                  }
                } else {
                  result[0] += -0.01660187494349542;
                }
              } else {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
                      result[0] += 0.01907309067646649;
                    } else {
                      result[0] += -0.12737326920117173;
                    }
                  } else {
                    if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.006384205570002958;
                    } else {
                      result[0] += 0.1114453640650333;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.285887241363526279) ) ) {
                    result[0] += -0.0009918117742391227;
                  } else {
                    result[0] += -0.03037539940938735;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.654679536819458896) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.623839378356934482) ) ) {
                  if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.868834793567657693) ) ) {
                    result[0] += -0.00785835704223804;
                  } else {
                    result[0] += -0.1460037066581029;
                  }
                } else {
                  if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.129040718078614169) ) ) {
                      result[0] += -0.12421136802669369;
                    } else {
                      result[0] += -0.04376869045912764;
                    }
                  } else {
                    result[0] += -0.019677590247401414;
                  }
                }
              } else {
                result[0] += -0.001987996325749268;
              }
            }
          } else {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += 0.1216725828757095;
            } else {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.718933820724488193) ) ) {
                result[0] += -0.040970367643226854;
              } else {
                if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += -0.0525536298399811;
                } else {
                  result[0] += 0.016277633485308248;
                }
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.938867926597595659) ) ) {
        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
          result[0] += -0.0866680503012166;
        } else {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
              result[0] += -0.17071241518598598;
            } else {
              result[0] += 0.06651507080660665;
            }
          } else {
            result[0] += -0.030704556497000875;
          }
        }
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.342454433441162998) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.431901693344116655) ) ) {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)6.809154510498047763) ) ) {
                result[0] += 0.030373172358376162;
              } else {
                result[0] += 0.1499624279716458;
              }
            } else {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.636499762535095659) ) ) {
                          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.44381141662597834) ) ) {
                            result[0] += -0.03932602248436681;
                          } else {
                            result[0] += 0.05080304579077345;
                          }
                        } else {
                          result[0] += 0.04480620187926097;
                        }
                      } else {
                        result[0] += -0.009715405265250915;
                      }
                    } else {
                      if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                          result[0] += 0.07634205916797603;
                        } else {
                          result[0] += 0.0018821548881227876;
                        }
                      } else {
                        result[0] += -0.027390797487663928;
                      }
                    }
                  } else {
                    result[0] += 0.06174566895402287;
                  }
                } else {
                  if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.283562898635865146) ) ) {
                    result[0] += -0.10219680588859806;
                  } else {
                    result[0] += 0.07258447703142142;
                  }
                }
              } else {
                result[0] += -0.022432967185094743;
              }
            }
          } else {
            result[0] += -0.008502230910125924;
          }
        } else {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.665476083755494052) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.262283086776734287) ) ) {
                result[0] += 0.03695906330824308;
              } else {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.023699979461724238;
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.071567356586456743) ) ) {
                    result[0] += 0.19839543428726544;
                  } else {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.465643882751465732) ) ) {
                        result[0] += -0.2055548679756568;
                      } else {
                        result[0] += -0.00618318249632906;
                      }
                    } else {
                      result[0] += 0.022898119788648;
                    }
                  }
                }
              }
            } else {
              result[0] += 0.036431790747202084;
            }
          } else {
            if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)7.508512496948243076) ) ) {
              result[0] += -0.010170535261753435;
            } else {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.1252460660340074;
              } else {
                result[0] += 0.004743038420544323;
              }
            }
          }
        }
      }
    }
  } else {
    result[0] += 0.00023457622059957638;
  }
  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
      if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.497866153717041238) ) ) {
        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
          if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)6.000000000000000888) ) ) {
            if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.051747083663941318) ) ) {
              result[0] += 0.006014776277161436;
            } else {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
                result[0] += 0.029179964084454592;
              } else {
                result[0] += 0.18271134131555392;
              }
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)14.04383373260498225) ) ) {
              if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.02952878096580915;
              } else {
                result[0] += -0.0015361371294520928;
              }
            } else {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.011246636632400369;
              } else {
                result[0] += 0.016646836694324117;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.868834793567657693) ) ) {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.855006217956543857) ) ) {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)4.331511974334717685) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)8.285748958587648261) ) ) {
                  result[0] += -0.0022079843279114073;
                } else {
                  if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.03420138359069913) ) ) {
                    result[0] += -0.015106547134522433;
                  } else {
                    result[0] += 0.1423335454126812;
                  }
                }
              } else {
                if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.962292194366455966) ) ) {
                  result[0] += -0.10195193340309955;
                } else {
                  result[0] += 0.06524780420415052;
                }
              }
            } else {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
                  result[0] += -0.015102841472537208;
                } else {
                  result[0] += -0.05565501714230098;
                }
              } else {
                result[0] += 0.02770450578092931;
              }
            }
          } else {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += -0.25448158999471987;
            } else {
              result[0] += -0.04862395950280962;
            }
          }
        }
      } else {
        result[0] += -0.02599047191067044;
      }
    } else {
      if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)2.500000000000000444) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.056097030639650214) ) ) {
          if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.917405366897583452) ) ) {
            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.868834793567657693) ) ) {
              result[0] += 0.0979079464952751;
            } else {
              result[0] += -0.03185898032875557;
            }
          } else {
            result[0] += 0.08735894679610207;
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.737386107444763628) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.07496595382690607) ) ) {
              result[0] += 0.10913976527258629;
            } else {
              result[0] += -0.04699498742181186;
            }
          } else {
            result[0] += -0.04858328034064962;
          }
        }
      } else {
        if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
            result[0] += 0.0005449915886944702;
          } else {
            if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += -0.004414679388799413;
            } else {
              result[0] += -0.0487689949547802;
            }
          }
        } else {
          result[0] += 0.0010583980195663497;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.624251961708069292) ) ) {
      result[0] += 0.00016564181723532993;
    } else {
      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
        if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.941534638404846635) ) ) {
          if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)12.00000000000000178) ) ) {
              result[0] += 0.025482634360721373;
            } else {
              result[0] += -0.009282797253711535;
            }
          } else {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += 0.011591765272224983;
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.48738741874694913) ) ) {
                result[0] += -0.011481941670547457;
              } else {
                result[0] += -0.06672736813155745;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)2.500000000000000444) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.241523027420044833) ) ) {
              result[0] += 0.09376280824191852;
            } else {
              result[0] += -0.0434833113237511;
            }
          } else {
            if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.895778417587281162) ) ) {
              if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.5240359306335467) ) ) {
                  if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)6.000000000000000888) ) ) {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += 0.005156856553277785;
                    } else {
                      if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.998158693313599077) ) ) {
                        result[0] += 0.0637022714588789;
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.403187274932863104) ) ) {
                          result[0] += -0.018629091862853008;
                        } else {
                          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.9054608345031756) ) ) {
                              result[0] += 0.059187574833470306;
                            } else {
                              result[0] += -0.01213307522759173;
                            }
                          } else {
                            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
                              if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                                result[0] += -0.03893110100055643;
                              } else {
                                result[0] += 0.052640605647986595;
                              }
                            } else {
                              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                                result[0] += 0.11086722424574146;
                              } else {
                                result[0] += -0.012342402080842328;
                              }
                            }
                          }
                        }
                      }
                    }
                  } else {
                    result[0] += 0.004043133114658391;
                  }
                } else {
                  result[0] += -0.015422324561874099;
                }
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.397998809814454013) ) ) {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.03420138359069913) ) ) {
                      if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.01634240150451749) ) ) {
                        result[0] += 0.04190311648083173;
                      } else {
                        result[0] += -0.056568033130359845;
                      }
                    } else {
                      result[0] += 0.03340285205958327;
                    }
                  } else {
                    result[0] += 1.4753952723839849e-05;
                  }
                } else {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.53326439857482999) ) ) {
                    if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
                      result[0] += -0.02848720668962175;
                    } else {
                      if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.510617971420288974) ) ) {
                          result[0] += 0.05880415964910825;
                        } else {
                          result[0] += -0.06952598427329965;
                        }
                      } else {
                        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.531673669815064365) ) ) {
                          result[0] += -0.015292489904081078;
                        } else {
                          result[0] += 0.10775093580611453;
                        }
                      }
                    }
                  } else {
                    result[0] += -0.0017092743691708085;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
                result[0] += -0.01828835840173173;
              } else {
                result[0] += -0.15649433981208438;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.07465314865112482) ) ) {
            result[0] += 0.03770308264259864;
          } else {
            result[0] += -0.025223626655984627;
          }
        } else {
          if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)7.329747915267945224) ) ) {
            result[0] += 0.015795673521211553;
          } else {
            result[0] += -0.034249335539937246;
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
    result[0] += 0.000545143724105797;
  } else {
    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.803987503051758701) ) ) {
      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
          if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.851041555404663974) ) ) {
                if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.114990711212159091) ) ) {
                  result[0] += 0.02570073126541213;
                } else {
                  result[0] += -0.010423829643753192;
                }
              } else {
                result[0] += -0.1283817339135451;
              }
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.497866153717041238) ) ) {
                if ( UNLIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.06764018105096008;
                } else {
                  if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                    result[0] += -0.019750837609622415;
                  } else {
                    result[0] += 0.04232731926081151;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.740319490432739702) ) ) {
                  result[0] += 0.0011792815166359113;
                } else {
                  result[0] += 0.027983323668541185;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.624251961708069292) ) ) {
              result[0] += 6.11013947697126e-05;
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                result[0] += -0.004343540304211185;
              } else {
                result[0] += -0.06871156864328556;
              }
            }
          }
        } else {
          result[0] += -0.0027685556853388775;
        }
      } else {
        if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
            result[0] += 0.01169519739349597;
          } else {
            if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.624251961708069292) ) ) {
              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += 0.0027341915936303337;
              } else {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.761470437049866167) ) ) {
                  result[0] += -0.013966469346049757;
                } else {
                  result[0] += 0.10351214893196124;
                }
              }
            } else {
              result[0] += -0.025522718367962383;
            }
          }
        } else {
          result[0] += -0.026088610509814626;
        }
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.921060562133789951) ) ) {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
            if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.56941866874694913) ) ) {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                    if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.472188472747804511) ) ) {
                      result[0] += 0.010077746063337742;
                    } else {
                      result[0] += 0.1044572772177652;
                    }
                  } else {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.537586927413940874) ) ) {
                      if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                        result[0] += -0.0156487126751211;
                      } else {
                        if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                            result[0] += -0.015337429495176894;
                          } else {
                            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                              if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.602003335952759233) ) ) {
                                result[0] += -0.028146916280070485;
                              } else {
                                if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                                  result[0] += 0.008135644731185294;
                                } else {
                                  result[0] += 0.05472644419094333;
                                }
                              }
                            } else {
                              result[0] += 0.009392100460106542;
                            }
                          }
                        } else {
                          if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            result[0] += 0.0038898476714433502;
                          } else {
                            if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)9.769521713256837714) ) ) {
                              result[0] += -0.02942490277224645;
                            } else {
                              result[0] += 0.07672877807877487;
                            }
                          }
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                        result[0] += -0.0850957193638532;
                      } else {
                        result[0] += -0.026971901967042973;
                      }
                    }
                  }
                } else {
                  result[0] += 0.004892373153736716;
                }
              } else {
                if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.816582441329956943) ) ) {
                    if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.283562898635865146) ) ) {
                      result[0] += -0.06998994413409843;
                    } else {
                      if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                        result[0] += -0.1046548652592933;
                      } else {
                        result[0] += 0.009796525237286378;
                      }
                    }
                  } else {
                    result[0] += -0.01985843701232949;
                  }
                } else {
                  if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.016408654344877347;
                  } else {
                    result[0] += -0.043349782520793384;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.067782521247864214) ) ) {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                  result[0] += 0.018383758171393618;
                } else {
                  result[0] += -0.006830756149115703;
                }
              } else {
                if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.141444921493531162) ) ) {
                    result[0] += 0.010955789950824947;
                  } else {
                    result[0] += 0.04801440323393933;
                  }
                } else {
                  result[0] += 0.0007211903615803885;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)1.151292562484741433) ) ) {
              if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += -0.010060893471918972;
              } else {
                result[0] += 0.04734592778578721;
              }
            } else {
              if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)9.554603576660158026) ) ) {
                result[0] += 0.003937689765823986;
              } else {
                result[0] += -0.009548271626625245;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.384246587753296343) ) ) {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.56941866874694913) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.153024196624756748) ) ) {
                if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.0030222956806028605;
                } else {
                  result[0] += 0.02566195292559456;
                }
              } else {
                if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)1.151292562484741433) ) ) {
                  result[0] += 0.039028585316818624;
                } else {
                  result[0] += -0.008120711660576497;
                }
              }
            } else {
              result[0] += 0.025999246801423168;
            }
          } else {
            if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
              if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.029386832269594616;
              } else {
                if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.03242942571217008;
                } else {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.637949228286744052) ) ) {
                    result[0] += 0.0010078624579733049;
                  } else {
                    result[0] += 0.06948321129985134;
                  }
                }
              }
            } else {
              result[0] += -0.0502283515349425;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.783891201019287998) ) ) {
          result[0] += -0.0007710344185569503;
        } else {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.088880300521851474) ) ) {
            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += -0.03081076298144323;
            } else {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += 0.011249028166600101;
              } else {
                result[0] += -0.03214123232573475;
              }
            }
          } else {
            result[0] += -0.0013926271440356513;
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.357691764831543413) ) ) {
        result[0] += -0.020061705597860843;
      } else {
        if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.700598716735840066) ) ) {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.909254074096680576) ) ) {
              result[0] += 0.004305609398103339;
            } else {
              result[0] += -0.16560732189484711;
            }
          } else {
            if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.009714544893800406;
              } else {
                result[0] += 0.14586692873748555;
              }
            } else {
              if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += -0.041454244058041004;
              } else {
                result[0] += 0.04815364103894197;
              }
            }
          }
        } else {
          result[0] += 0.004955886551875989;
        }
      }
    } else {
      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.329314231872559482) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.205624103546144354) ) ) {
          if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
            if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.497866153717041238) ) ) {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                  result[0] += -0.03967677792677593;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.941167116165162021) ) ) {
                    result[0] += 0.016323570809311832;
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.594915628433228427) ) ) {
                      if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
                        result[0] += 0.00500710337066777;
                      } else {
                        if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                            if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.198464870452881303) ) ) {
                              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.898905277252199042) ) ) {
                                result[0] += 0.054762426547394584;
                              } else {
                                result[0] += -0.017565372489157117;
                              }
                            } else {
                              result[0] += 0.10680766865210517;
                            }
                          } else {
                            result[0] += -0.038238615434683376;
                          }
                        } else {
                          if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.650573849678039995) ) ) {
                            result[0] += -0.04474533805384793;
                          } else {
                            result[0] += 0.06089562457118708;
                          }
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.534971714019776279) ) ) {
                            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.071567356586456743) ) ) {
                              if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                                result[0] += -0.12493547634694394;
                              } else {
                                result[0] += -0.014386500103109929;
                              }
                            } else {
                              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.247576236724854404) ) ) {
                                if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.029068946838379794) ) ) {
                                    result[0] += 0.07053296731164956;
                                  } else {
                                    result[0] += 0.013862867030090532;
                                  }
                                } else {
                                  if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.350240230560303178) ) ) {
                                    result[0] += -0.11002196107367346;
                                  } else {
                                    result[0] += -0.005623502881263506;
                                  }
                                }
                              } else {
                                if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                                  result[0] += 0.0016590006283004302;
                                } else {
                                  result[0] += -0.024784704514805314;
                                }
                              }
                            }
                          } else {
                            result[0] += -0.04003596077862997;
                          }
                        } else {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.347943067550660068) ) ) {
                            result[0] += 0.07131344706496208;
                          } else {
                            if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2565.000000000000455) ) ) {
                              result[0] += 0.01428734381309278;
                            } else {
                              result[0] += -0.06446442477928904;
                            }
                          }
                        }
                      } else {
                        result[0] += 0.01414941202680225;
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.94957673549652144) ) ) {
                  result[0] += -0.029132839952739062;
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.94957673549652144) ) ) {
                    result[0] += 0.04260797588168475;
                  } else {
                    if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += 0.0012758753584491238;
                    } else {
                      result[0] += -0.051313084915616786;
                    }
                  }
                }
              }
            } else {
              result[0] += -0.04936776002213379;
            }
          } else {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += 0.08337778259612022;
            } else {
              if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.917405366897583452) ) ) {
                result[0] += -0.00160360194759018;
              } else {
                result[0] += -0.017245356222753656;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
              result[0] += -0.11996340034760772;
            } else {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)7.971558809280396396) ) ) {
                result[0] += -0.00815548864854027;
              } else {
                result[0] += -0.19503182535885644;
              }
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.637949228286744052) ) ) {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += 0.08328229029198218;
              } else {
                result[0] += 0.0019104960625237562;
              }
            } else {
              result[0] += -0.008939563963670736;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.909102678298951083) ) ) {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.868834793567657693) ) ) {
              result[0] += 0.022571869849972292;
            } else {
              result[0] += -0.041432522016715885;
            }
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.53326439857482999) ) ) {
              result[0] += 0.016799879717040687;
            } else {
              if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)6.588238239288330966) ) ) {
                result[0] += 0.08276976678365341;
              } else {
                if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
                  result[0] += -0.23925027261491888;
                } else {
                  result[0] += 0.059022208163955474;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.780892848968506748) ) ) {
              if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
                if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.012675821781158891) ) ) {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += -0.050452086749919495;
                  } else {
                    result[0] += 0.0024762592330576007;
                  }
                } else {
                  result[0] += 0.0034651564610325702;
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)1.497866153717041238) ) ) {
                  result[0] += -0.13499115585038085;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.605120182037354404) ) ) {
                    result[0] += 0.022936111074222783;
                  } else {
                    result[0] += -0.01088882030023966;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                  if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.138731002807618076) ) ) {
                    result[0] += -0.028144786726245503;
                  } else {
                    result[0] += 0.013126741582617556;
                  }
                } else {
                  result[0] += -0.14008021844827914;
                }
              } else {
                result[0] += -0.03851476268945722;
              }
            }
          } else {
            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.497866153717041238) ) ) {
              result[0] += 0.0027267592191647897;
            } else {
              result[0] += -0.02441174914029962;
            }
          }
        }
      }
    }
  } else {
    result[0] += 0.00021108640708361457;
  }
  if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
    if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)2.567899227142334428) ) ) {
      if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
        result[0] += 0.0007683596992362369;
      } else {
        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.700598716735840066) ) ) {
          result[0] += 0.03207281689787687;
        } else {
          result[0] += 0.009382901355105987;
        }
      }
    } else {
      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.992907285690308505) ) ) {
        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
          if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += -0.030640369299563798;
          } else {
            if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.06057546961937347;
            } else {
              if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.011443605303779353;
              } else {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.052836954724913524;
                } else {
                  result[0] += 0.0160183912819656;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += 0.0015504852376658459;
            } else {
              if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.006768739860587349;
                  } else {
                    result[0] += -0.018793197170989664;
                  }
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.665476083755494052) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.810334205627442294) ) ) {
                      result[0] += 0.005854543719846303;
                    } else {
                      result[0] += -0.04379533290617384;
                    }
                  } else {
                    result[0] += -0.053442485635336046;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.602003335952759233) ) ) {
                  result[0] += -0.00846311489954599;
                } else {
                  result[0] += 0.009934723410651432;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
              if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.909855604171753818) ) ) {
                  result[0] += 0.0019341633452751095;
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.403187274932863104) ) ) {
                    result[0] += -0.0069152152435845915;
                  } else {
                    result[0] += -0.03660455110063364;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.53326439857482999) ) ) {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += 0.07343249782880777;
                  } else {
                    result[0] += -0.017110908669412693;
                  }
                } else {
                  if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                    result[0] += 0.02704984996304424;
                  } else {
                    result[0] += 0.09028542759587933;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.329314231872559482) ) ) {
                    result[0] += -0.002825190299398223;
                  } else {
                    result[0] += -0.07838423541616887;
                  }
                } else {
                  if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += 0.01165526635473463;
                  } else {
                    result[0] += -0.03849566905697443;
                  }
                }
              } else {
                if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
                      if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.020127415657043901) ) ) {
                        result[0] += 0.002508229996762436;
                      } else {
                        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                          if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.935661792755128729) ) ) {
                              if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                                result[0] += -0.028601437630813423;
                              } else {
                                result[0] += 0.005112804967910327;
                              }
                            } else {
                              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.778982400894165927) ) ) {
                                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                                  result[0] += -0.0009375364967184278;
                                } else {
                                  result[0] += -0.046556845064039606;
                                }
                              } else {
                                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                                  if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                                    result[0] += -0.015470183083939722;
                                  } else {
                                    if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                                      result[0] += 0.005226148366568537;
                                    } else {
                                      if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                                        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                                          result[0] += -0.05599985671121663;
                                        } else {
                                          result[0] += 0.04166453430864353;
                                        }
                                      } else {
                                        result[0] += 0.07325770357953834;
                                      }
                                    }
                                  }
                                } else {
                                  if ( UNLIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.241523027420044833) ) ) {
                                      result[0] += -0.052876790203957885;
                                    } else {
                                      result[0] += 0.07924376198215682;
                                    }
                                  } else {
                                    result[0] += -0.060283832334770485;
                                  }
                                }
                              }
                            }
                          } else {
                            result[0] += -0.01957015335040206;
                          }
                        } else {
                          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.23832273483276456) ) ) {
                            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.142630577087403232) ) ) {
                              result[0] += 0.0057838528918750295;
                            } else {
                              if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                                result[0] += -0.04861792800747604;
                              } else {
                                result[0] += 0.021747726746830835;
                              }
                            }
                          } else {
                            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                                result[0] += 0.027903792074434303;
                              } else {
                                result[0] += -0.060695607505110584;
                              }
                            } else {
                              result[0] += -0.052862712564165505;
                            }
                          }
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                        result[0] += -0.08407924741877279;
                      } else {
                        if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)9.012558460235597479) ) ) {
                          result[0] += -0.04704080048210149;
                        } else {
                          result[0] += 0.056379840959240646;
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.433569431304932529) ) ) {
                      result[0] += 0.004040816190814627;
                    } else {
                      result[0] += -0.06366228028067404;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                    if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.357691764831543413) ) ) {
                      result[0] += -0.073171630073446;
                    } else {
                      result[0] += 0.02912872651455586;
                    }
                  } else {
                    result[0] += -0.008085815056626195;
                  }
                }
              }
            }
          }
        }
      } else {
        result[0] += -0.009703989717266208;
      }
    }
  } else {
    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.510617971420288974) ) ) {
      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.242453336715698464) ) ) {
            result[0] += -0.07947586789855253;
          } else {
            result[0] += 0.006042506259062033;
          }
        } else {
          result[0] += -0.022504012547234345;
        }
      } else {
        result[0] += -0.005362961841269002;
      }
    } else {
      if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.182021141052246982) ) ) {
        result[0] += 0.0005313319843342745;
      } else {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
          result[0] += 0.008685722744162147;
        } else {
          result[0] += -0.008865281421537002;
        }
      }
    }
  }
  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.959391355514527255) ) ) {
        result[0] += -0.0010264357879521205;
      } else {
        result[0] += -0.012781089312444274;
      }
    } else {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
        result[0] += -0.0018984645739329462;
      } else {
        if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
          if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.088880300521851474) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.242453336715698464) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)16.13862895965576527) ) ) {
                  result[0] += -0.04977434959214282;
                } else {
                  result[0] += 0.0730257290423487;
                }
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.025192260742188388) ) ) {
                  result[0] += 0.007110415626434385;
                } else {
                  if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.537586927413940874) ) ) {
                      result[0] += -0.015037841511563808;
                    } else {
                      result[0] += -0.08001635899545953;
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.643222332000734198) ) ) {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
                        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.772694945335388628) ) ) {
                          if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                            result[0] += 0.022896683515551016;
                          } else {
                            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                              result[0] += -0.00447543760313579;
                            } else {
                              result[0] += -0.10371290631578958;
                            }
                          }
                        } else {
                          result[0] += 0.04264118579691854;
                        }
                      } else {
                        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                          result[0] += 0.010330607498391294;
                        } else {
                          result[0] += -0.023897411858215607;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
                        result[0] += -0.017387106101437868;
                      } else {
                        result[0] += 0.008231794246319367;
                      }
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.556798219680787021) ) ) {
                  result[0] += 0.027381843500193216;
                } else {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                    result[0] += -0.013558264071235174;
                  } else {
                    result[0] += -0.04486018645414189;
                  }
                }
              } else {
                result[0] += 0.005688509393132167;
              }
            }
          } else {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += 0.09699096652120119;
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.602003335952759233) ) ) {
                    result[0] += -0.03101622361451454;
                  } else {
                    result[0] += -0.11669428558882947;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.36986422538757413) ) ) {
                  if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.552201986312867099) ) ) {
                        result[0] += -0.043998972533126984;
                      } else {
                        result[0] += 0.03541106639754948;
                      }
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.510617971420288974) ) ) {
                        result[0] += -0.0747959984722872;
                      } else {
                        result[0] += 0.0053873268423990685;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.53326439857482999) ) ) {
                      result[0] += -0.02359962254476149;
                    } else {
                      result[0] += 0.04595787293736828;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.637949228286744052) ) ) {
                    result[0] += 0.007590063260635685;
                  } else {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                      if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)0.8958797454833985485) ) ) {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.837713479995728427) ) ) {
                          result[0] += 0.05111125678894634;
                        } else {
                          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.637949228286744052) ) ) {
                              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.353313446044923651) ) ) {
                                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                                  result[0] += -0.13728302894304273;
                                } else {
                                  result[0] += -0.03445530719204058;
                                }
                              } else {
                                result[0] += 0.007549632043307601;
                              }
                            } else {
                              result[0] += 0.01150475865141569;
                            }
                          } else {
                            result[0] += 0.029388030734098243;
                          }
                        }
                      } else {
                        result[0] += -0.04749404302645071;
                      }
                    } else {
                      if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.36105370521545499) ) ) {
                          result[0] += 0.03933046643075963;
                        } else {
                          result[0] += 0.08771507719216608;
                        }
                      } else {
                        result[0] += -0.01152362032429279;
                      }
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += 0.11540127160552519;
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.918272972106934482) ) ) {
                  result[0] += 0.007697676309960596;
                } else {
                  if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
                      result[0] += 0.077490395147579;
                    } else {
                      result[0] += 0.004194476824489973;
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.152389049530031073) ) ) {
                      result[0] += 0.01767776072657019;
                    } else {
                      if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += 0.1006573264922016;
                      } else {
                        result[0] += 0.038984451529567005;
                      }
                    }
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.35441589355468928) ) ) {
            if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.909254074096680576) ) ) {
                result[0] += 0.008172474827588618;
              } else {
                result[0] += -0.028185834198775003;
              }
            } else {
              if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
                  result[0] += -0.0052290988384549675;
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.450390577316285068) ) ) {
                    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += -0.05595792410381656;
                    } else {
                      result[0] += 0.004669457361102759;
                    }
                  } else {
                    result[0] += 0.014270642670291087;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.433652400970459873) ) ) {
                    result[0] += -0.05794690650505918;
                  } else {
                    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += -0.05305766419265027;
                    } else {
                      result[0] += 0.01973324824101609;
                    }
                  }
                } else {
                  result[0] += 0.07890303286458997;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
              result[0] += 0.004191474246607282;
            } else {
              result[0] += 0.05008520286249329;
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)1.777674019336700661) ) ) {
      if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
        result[0] += 0.032264677031984455;
      } else {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
          result[0] += -0.08441382256116203;
        } else {
          result[0] += 0.008387895225862653;
        }
      }
    } else {
      result[0] += 0.0007288260649283981;
    }
  }
  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.179772853851319248) ) ) {
      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.872538805007935458) ) ) {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.572941064834595615) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.205624103546144354) ) ) {
            result[0] += 0.0014765798261956679;
          } else {
            if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)6.000000000000000888) ) ) {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += 0.002943326305268959;
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
                  result[0] += 0.13516933692285776;
                } else {
                  result[0] += 0.02232010766518031;
                }
              }
            } else {
              result[0] += -0.004591098343599087;
            }
          }
        } else {
          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
            result[0] += -0.006943900764456501;
          } else {
            result[0] += -0.04181987567837842;
          }
        }
      } else {
        result[0] += 0.07436235719854643;
      }
    } else {
      if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
        result[0] += 0.006975815316343919;
      } else {
        if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.986918687820435458) ) ) {
            result[0] += -0.010924935427924325;
          } else {
            result[0] += 0.07278571861094761;
          }
        } else {
          result[0] += -0.033616177332036916;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
      if ( LIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.276966691017151323) ) ) {
          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
            result[0] += -0.022094463509804194;
          } else {
            result[0] += -0.07988863440907318;
          }
        } else {
          result[0] += 0.0003242275773027033;
        }
      } else {
        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.44140100479126021) ) ) {
          result[0] += 0.12082615117647531;
        } else {
          result[0] += 0.010023263545669531;
        }
      }
    } else {
      if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.605039834976196733) ) ) {
            if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.023093045967784886;
            } else {
              result[0] += -0.04782914123662471;
            }
          } else {
            if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
              if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.892608642578125888) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.464211463928224433) ) ) {
                    result[0] += -0.006728657228742297;
                  } else {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.384246587753296343) ) ) {
                      result[0] += 0.13904832270056092;
                    } else {
                      result[0] += 0.030091434719787827;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.664408206939698154) ) ) {
                    result[0] += 0.01785505922216561;
                  } else {
                    result[0] += 0.13387310305266034;
                  }
                }
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.084203958511353427) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.947818994522095615) ) ) {
                    result[0] += 0.010649550500647822;
                  } else {
                    result[0] += 0.08458815305980123;
                  }
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.643222332000734198) ) ) {
                    result[0] += -0.06740562073631337;
                  } else {
                    result[0] += 0.049245326317811616;
                  }
                }
              }
            } else {
              result[0] += 0.056403598234928076;
            }
          }
        } else {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.35306882858276456) ) ) {
            result[0] += 0.000845078552712051;
          } else {
            result[0] += 0.02019386345152792;
          }
        }
      } else {
        if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)3.000000000000000444) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.028861761093140537) ) ) {
              if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
                  result[0] += -0.07626048513764766;
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.553712725639343706) ) ) {
                    if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                      result[0] += -0.0005681232857128181;
                    } else {
                      result[0] += 0.2474068122170366;
                    }
                  } else {
                    if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.467917680740357333) ) ) {
                        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.540854334831238237) ) ) {
                          result[0] += -0.06416792089758633;
                        } else {
                          result[0] += 0.014970776694442689;
                        }
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.241523027420044833) ) ) {
                          result[0] += -0.02064266305023042;
                        } else {
                          result[0] += 0.023396635021742652;
                        }
                      }
                    } else {
                      result[0] += -0.06605063740051703;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.883387088775636542) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                    result[0] += 0.03151719571865787;
                  } else {
                    result[0] += 0.0070272390525292884;
                  }
                } else {
                  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.008193613539571284;
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.154959201812744585) ) ) {
                      result[0] += 0.016928440203854086;
                    } else {
                      if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += -0.01250962282855179;
                      } else {
                        result[0] += -0.05218528133769276;
                      }
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += -0.04408107743471648;
              } else {
                result[0] += 0.12341851098553878;
              }
            }
          } else {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)46.00000000000000711) ) ) {
              result[0] += -0.07917926322989934;
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.357691764831543413) ) ) {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                  if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.0732494071564233;
                  } else {
                    result[0] += 0.011972404002994357;
                  }
                } else {
                  result[0] += -0.04695885470316869;
                }
              } else {
                if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.00538769288217179;
                } else {
                  if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
                    if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.013174245797532867;
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.056097030639650214) ) ) {
                        result[0] += -0.01591704017357625;
                      } else {
                        result[0] += -0.08922212679734835;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.126503944396974433) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.241523027420044833) ) ) {
                        result[0] += -0.13494474870117687;
                      } else {
                        result[0] += -0.02262254028588078;
                      }
                    } else {
                      if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += -0.04249965467801368;
                      } else {
                        result[0] += 0.03560532195814902;
                      }
                    }
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)16.83253097534180043) ) ) {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)4.125962495803833896) ) ) {
                result[0] += 0.002269608356433915;
              } else {
                result[0] += 0.03526481638666549;
              }
            } else {
              result[0] += 0.049854399985429676;
            }
          } else {
            result[0] += 3.040917379820703e-05;
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
    result[0] += 0.0005333821088090195;
  } else {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.439304351806642401) ) ) {
      if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
        if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
          if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.58491539955139249) ) ) {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                result[0] += -0.018721173627328617;
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.219419956207276279) ) ) {
                  result[0] += -0.01474723795189918;
                } else {
                  result[0] += 0.011691546714336478;
                }
              }
            } else {
              if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)24.00000000000000355) ) ) {
                if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += 0.04592641230597155;
                } else {
                  if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.449861526489258257) ) ) {
                      result[0] += 0.00672738276264629;
                    } else {
                      result[0] += -0.012401326895953734;
                    }
                  } else {
                    result[0] += -0.007717150208316061;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.53326439857482999) ) ) {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                    if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)4.166635274887085849) ) ) {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.861792564392090288) ) ) {
                        result[0] += -0.05083504944313109;
                      } else {
                        result[0] += -0.013265191885783543;
                      }
                    } else {
                      result[0] += -0.0013687676494429804;
                    }
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.914472818374634233) ) ) {
                      result[0] += -0.0405874423239826;
                    } else {
                      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.540854334831238237) ) ) {
                        result[0] += 0.01257216280074078;
                      } else {
                        result[0] += 0.052661254522054404;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.450390577316285068) ) ) {
                        result[0] += 0.007936141199052688;
                      } else {
                        result[0] += 0.02656038105902017;
                      }
                    } else {
                      if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                          result[0] += 0.010850172207582455;
                        } else {
                          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.750972747802735263) ) ) {
                              result[0] += 0.005483708796065719;
                            } else {
                              result[0] += -0.06180966198837909;
                            }
                          } else {
                            if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                              result[0] += -0.0489490653042752;
                            } else {
                              result[0] += 0.02247217502410322;
                            }
                          }
                        }
                      } else {
                        result[0] += -0.03186410125359904;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.947818994522095615) ) ) {
                      if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.673553824424744096) ) ) {
                        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.597137451171875888) ) ) {
                            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.241523027420044833) ) ) {
                              result[0] += -0.014775604767072537;
                            } else {
                              if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                                result[0] += -0.05014324296707683;
                              } else {
                                result[0] += -0.20344454359223466;
                              }
                            }
                          } else {
                            result[0] += 0.0131745503727514;
                          }
                        } else {
                          result[0] += 0.01319881973812636;
                        }
                      } else {
                        result[0] += 0.019516773847987515;
                      }
                    } else {
                      result[0] += 0.03238915379381427;
                    }
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.36986422538757413) ) ) {
              result[0] += 0.001356361888792126;
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.53326439857482999) ) ) {
                result[0] += -0.01835287361192482;
              } else {
                result[0] += -0.07399130063979613;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.48738741874694913) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.918272972106934482) ) ) {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.700598716735840066) ) ) {
                  result[0] += 0.009604569731777082;
                } else {
                  result[0] += -0.03268611229943616;
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.58491539955139249) ) ) {
                  result[0] += 0.008241320182250728;
                } else {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.285887241363526279) ) ) {
                      result[0] += -0.0026074840055701036;
                    } else {
                      result[0] += -0.02105007412526548;
                    }
                  } else {
                    if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.0642563915274851;
                    } else {
                      result[0] += -0.011550912496963148;
                    }
                  }
                }
              }
            } else {
              result[0] += 0.015103009915395735;
            }
          } else {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.30853915214538663) ) ) {
                result[0] += 0.0017116168975762327;
              } else {
                result[0] += -0.021207751622126736;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.97887301445007413) ) ) {
                result[0] += -0.016828383197540788;
              } else {
                result[0] += -0.049427328926489864;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.743881702423096591) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.189540147781372958) ) ) {
              result[0] += -0.0016485840223464225;
            } else {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += 0.018651066137569377;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.556798219680787021) ) ) {
                    result[0] += -0.03632121569139476;
                  } else {
                    result[0] += 0.008757377758949919;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.766185760498047763) ) ) {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.700598716735840066) ) ) {
                    result[0] += 0.050374169509797506;
                  } else {
                    result[0] += 0.015205260142820354;
                  }
                } else {
                  if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += 0.09667434824968935;
                  } else {
                    result[0] += -0.01085337619183119;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.288254261016846591) ) ) {
                result[0] += -0.002208347369355286;
              } else {
                result[0] += -0.04167813662735841;
              }
            } else {
              result[0] += 0.005106081301950934;
            }
          }
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.511434078216553178) ) ) {
              result[0] += 0.003287001872050522;
            } else {
              result[0] += -0.02775885512448345;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.556798219680787021) ) ) {
              result[0] += 0.04517086897499223;
            } else {
              result[0] += -0.009434079640908744;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
        result[0] += -0.004384298057092541;
      } else {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.507949829101563388) ) ) {
          if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
            result[0] += -0.0043992540837217075;
          } else {
            result[0] += -0.021021915882250858;
          }
        } else {
          result[0] += 0.04067102109773065;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)17.27274703979492543) ) ) {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.174569487571716753) ) ) {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.426736354827881748) ) ) {
          result[0] += 0.0010142459383365185;
        } else {
          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
            result[0] += -0.00944820663465052;
          } else {
            result[0] += 0.03551075887111829;
          }
        }
      } else {
        result[0] += -0.008531608599405957;
      }
    } else {
      if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
        result[0] += 0.019287106501564986;
      } else {
        if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)2.012675821781158891) ) ) {
          result[0] += 0.025970419941273738;
        } else {
          result[0] += 0.1667865604773167;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.13022470474243342) ) ) {
      if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
        result[0] += 0.0004531901548114591;
      } else {
        if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.921924352645874468) ) ) {
            if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
              result[0] += 0.021207671804214227;
            } else {
              result[0] += -0.01376572217023382;
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.321723937988282138) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.105651378631592685) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.041387319564820224) ) ) {
                  result[0] += 0.03982762447240229;
                } else {
                  result[0] += -0.01599759435783068;
                }
              } else {
                result[0] += -0.02686059922365838;
              }
            } else {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += 0.02015311424282408;
              } else {
                if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)6.170655488967896396) ) ) {
                  result[0] += -0.022592608870423372;
                } else {
                  result[0] += -0.11072867162278484;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.23832273483276456) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.288254261016846591) ) ) {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.142630577087403232) ) ) {
                  result[0] += -0.11426505463757666;
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.379217386245728427) ) ) {
                    result[0] += 0.07964872959331165;
                  } else {
                    result[0] += -0.03466556588414399;
                  }
                }
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.216319084167481357) ) ) {
                  result[0] += 0.056989044700768134;
                } else {
                  result[0] += -0.03692423234987643;
                }
              }
            } else {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.745876312255860263) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.659039497375490058) ) ) {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.531673669815064365) ) ) {
                      result[0] += 0.06667235909853031;
                    } else {
                      result[0] += 0.00983032780210866;
                    }
                  } else {
                    result[0] += -0.011060553111288887;
                  }
                } else {
                  result[0] += -0.048625669072648786;
                }
              } else {
                result[0] += -0.039946526521127367;
              }
            }
          } else {
            result[0] += 0.024906071074069974;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
        if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.63218307495117365) ) ) {
              result[0] += 0.06770259163023229;
            } else {
              result[0] += -0.02846468302596086;
            }
          } else {
            result[0] += -0.006866059235770003;
          }
        } else {
          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)2.393745899200439897) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.53326439857482999) ) ) {
                if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += 0.00019060960817012174;
                } else {
                  result[0] += -0.026876577939560588;
                }
              } else {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                  if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)7.211187601089478427) ) ) {
                    if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)7.11963534355163663) ) ) {
                      if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)7.030415296554566318) ) ) {
                        result[0] += -0.012276698758189136;
                      } else {
                        result[0] += -0.16036081597683224;
                      }
                    } else {
                      result[0] += 0.0920421951788402;
                    }
                  } else {
                    result[0] += -0.10858252498830807;
                  }
                } else {
                  result[0] += 0.008731992472785065;
                }
              }
            } else {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.617236852645874912) ) ) {
                result[0] += 0.007994889991627998;
              } else {
                result[0] += -0.11519251239960318;
              }
            }
          } else {
            result[0] += 0.09759418784528709;
          }
        }
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
            if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
              if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                result[0] += -0.06440410316588895;
              } else {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                  result[0] += -0.0334444704884904;
                } else {
                  result[0] += -0.002104239507031972;
                }
              }
            } else {
              if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.004873958506965769;
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.507949829101563388) ) ) {
                  result[0] += -0.03775335182336359;
                } else {
                  result[0] += 0.050754772212894074;
                }
              }
            }
          } else {
            result[0] += -0.03714388364072167;
          }
        } else {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.909855604171753818) ) ) {
            if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)14.41290044784546076) ) ) {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += -0.0702836323874823;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.16594791412353693) ) ) {
                      result[0] += -0.17041837284872974;
                    } else {
                      result[0] += -0.007121853779897558;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.0026789338415059145;
                  } else {
                    if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += -0.023605874760608853;
                    } else {
                      result[0] += 0.0324967403313299;
                    }
                  }
                }
              } else {
                result[0] += -0.019609233448505528;
              }
            } else {
              if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.725620865821838823) ) ) {
                  result[0] += 0.01697145638195503;
                } else {
                  result[0] += -0.01676217495935691;
                }
              } else {
                if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.0052613284834653875;
                } else {
                  result[0] += -0.03340677746313103;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += -0.01795171963287528;
              } else {
                if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += 0.021418407339385232;
                } else {
                  result[0] += -0.06598952235049234;
                }
              }
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.56941866874694913) ) ) {
                result[0] += 0.027765903686168082;
              } else {
                result[0] += 0.09477215203651468;
              }
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)1.497866153717041238) ) ) {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.962127923965454546) ) ) {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.426736354827881748) ) ) {
          result[0] += 0.0008493113175997883;
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.242078304290772373) ) ) {
            if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += -0.0022453320260858996;
            } else {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.597323656082154208) ) ) {
                  result[0] += -0.053923865546427585;
                } else {
                  result[0] += -0.00540616950438775;
                }
              } else {
                result[0] += 0.008393002854220998;
              }
            }
          } else {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
              if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.01022206765704384;
              } else {
                result[0] += -0.10739747525163734;
              }
            } else {
              if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += -0.08359471684979206;
              } else {
                result[0] += 0.00799831054197213;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.848652839660646308) ) ) {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.531673669815064365) ) ) {
            result[0] += -0.008049695981369403;
          } else {
            result[0] += -0.04632448609280492;
          }
        } else {
          result[0] += -0.0025583506595149986;
        }
      }
    } else {
      result[0] += 0.03155717614520805;
    }
  } else {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.126503944396974433) ) ) {
      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.164715528488160068) ) ) {
        if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
          if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.778982400894165927) ) ) {
              result[0] += -0.03215723366334441;
            } else {
              result[0] += 0.03953197559631084;
            }
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.357691764831543413) ) ) {
              result[0] += -0.014015295045415592;
            } else {
              result[0] += 0.0009438019440311221;
            }
          }
        } else {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.921924352645874468) ) ) {
                result[0] += 0.00452585829727824;
              } else {
                result[0] += -0.03241144465003936;
              }
            } else {
              result[0] += 0.011947380648295267;
            }
          } else {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
              if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.013655528246557408;
              } else {
                if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)2.962127923965454546) ) ) {
                  if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                    result[0] += -0.05630583927599713;
                  } else {
                    result[0] += -0.015237485476943502;
                  }
                } else {
                  if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += 0.007079250184198846;
                  } else {
                    result[0] += -0.027508531124305197;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.700598716735840066) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.97887301445007413) ) ) {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.467917680740357333) ) ) {
                    if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)2.012675821781158891) ) ) {
                      result[0] += 0.006992477965625712;
                    } else {
                      result[0] += 0.25416253158212804;
                    }
                  } else {
                    result[0] += 0.042804586199398025;
                  }
                } else {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += 0.009598598177201288;
                  } else {
                    result[0] += -0.023614058335852277;
                  }
                }
              } else {
                result[0] += 0.05507456118251393;
              }
            }
          }
        }
      } else {
        result[0] += -0.005168169144639786;
      }
    } else {
      if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                result[0] += -0.04791720024249873;
              } else {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                  result[0] += -0.09075447966173951;
                } else {
                  result[0] += -0.0008647347119796232;
                }
              }
            } else {
              result[0] += -0.030494366264090158;
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.938867926597595659) ) ) {
              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += -0.002611190594255155;
              } else {
                result[0] += 0.13707470872460842;
              }
            } else {
              result[0] += -0.005836314543546382;
            }
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.342454433441162998) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.531673669815064365) ) ) {
              result[0] += -0.0005386049506539588;
            } else {
              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.0012786077938763839;
              } else {
                result[0] += -0.04346504753030819;
              }
            }
          } else {
            if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += -0.0004223648914334019;
            } else {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.467917680740357333) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.802696108818054643) ) ) {
                  result[0] += 0.027413972380405507;
                } else {
                  if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += 0.0125818872759368;
                  } else {
                    result[0] += -0.015754438650040257;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                    if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.04367167873939879;
                    } else {
                      result[0] += 0.014321976672114899;
                    }
                  } else {
                    result[0] += 0.03359335192060292;
                  }
                } else {
                  result[0] += 0.03155433780712329;
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.0014594506560939586;
            } else {
              result[0] += -0.02981514840202647;
            }
          } else {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.511434078216553178) ) ) {
              result[0] += 0.04760145109876915;
            } else {
              result[0] += 0.007860155212399108;
            }
          }
        } else {
          if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.142630577087403232) ) ) {
                result[0] += -0.02941622695069725;
              } else {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                  result[0] += -0.03849610441062715;
                } else {
                  if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)2.138333082199097124) ) ) {
                    result[0] += 0.003293836055271754;
                  } else {
                    if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += -0.0006277347091183965;
                    } else {
                      result[0] += -0.10861373988662376;
                    }
                  }
                }
              }
            } else {
              result[0] += -0.03293586770841122;
            }
          } else {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)12.00000000000000178) ) ) {
                result[0] += -0.023834636276493112;
              } else {
                result[0] += -0.06463983333652229;
              }
            } else {
              result[0] += -0.06701298516127283;
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
      if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.41263532638549982) ) ) {
          if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
            if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
              result[0] += 0.003808227070783558;
            } else {
              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.524927973747253862) ) ) {
                result[0] += -0.0850394742469768;
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.921060562133789951) ) ) {
                  result[0] += -0.03156125794354307;
                } else {
                  result[0] += 0.06992356865408424;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.126503944396974433) ) ) {
              if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.602003335952759233) ) ) {
                result[0] += -0.043148902601768686;
              } else {
                result[0] += 0.08335293255976045;
              }
            } else {
              result[0] += -0.07207949120167387;
            }
          }
        } else {
          if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.774904012680054599) ) ) {
            if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.051747083663941318) ) ) {
              result[0] += -0.11332325488084385;
            } else {
              result[0] += 0.0031187840193579817;
            }
          } else {
            result[0] += -0.18326041248657354;
          }
        }
      } else {
        if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)3.000000000000000444) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.59476566314697443) ) ) {
            result[0] += 0.0007453224716098771;
          } else {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += 0.04284459532630962;
            } else {
              result[0] += 0.0070619510071459855;
            }
          }
        } else {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.511434078216553178) ) ) {
              result[0] += -0.02777566751262063;
            } else {
              if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.675038576126099521) ) ) {
                  result[0] += 0.011081428837392535;
                } else {
                  result[0] += -0.011530955156057262;
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
                  result[0] += -0.13104744707190386;
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
                    result[0] += 0.11822938059772295;
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.634540319442749912) ) ) {
                      result[0] += -0.008224021357300609;
                    } else {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.81821727752685725) ) ) {
                        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.78735828399658381) ) ) {
                          result[0] += 0.031345997390635816;
                        } else {
                          result[0] += -0.030646278875825952;
                        }
                      } else {
                        result[0] += 0.06561255752450966;
                      }
                    }
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.673553824424744096) ) ) {
              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += 0.024777235889739825;
              } else {
                result[0] += -0.007237386109470063;
              }
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.174569487571716753) ) ) {
                if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)6.210597991943360263) ) ) {
                  result[0] += -0.009375309536196553;
                } else {
                  result[0] += -0.11867614536941085;
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.510617971420288974) ) ) {
                  result[0] += -0.004919487971831569;
                } else {
                  result[0] += 0.061866504467967744;
                }
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.58491539955139249) ) ) {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.467917680740357333) ) ) {
          result[0] += -0.009933094285204516;
        } else {
          result[0] += 0.02360455351065145;
        }
      } else {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
          result[0] += -0.013229795885523919;
        } else {
          result[0] += -0.05185246272916671;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.888826131820679155) ) ) {
        result[0] += -0.05152198631161751;
      } else {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.487163543701172763) ) ) {
          if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.676220536231995073) ) ) {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.637949228286744052) ) ) {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
                if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.449861526489258257) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.029068946838379794) ) ) {
                    result[0] += -0.01592897554738563;
                  } else {
                    if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.467917680740357333) ) ) {
                      result[0] += 0.08550343944604596;
                    } else {
                      result[0] += -0.013923345868544499;
                    }
                  }
                } else {
                  result[0] += 0.10524505401431128;
                }
              } else {
                if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.594915628433228427) ) ) {
                    result[0] += -0.048810086735921444;
                  } else {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.085941076278687412) ) ) {
                      result[0] += 0.07775030947539191;
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.778982400894165927) ) ) {
                        result[0] += -0.1019630479043036;
                      } else {
                        result[0] += 0.02042100917390602;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.96495962142944514) ) ) {
                      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.53326439857482999) ) ) {
                        result[0] += -0.038843688859102235;
                      } else {
                        result[0] += 0.04153814031981846;
                      }
                    } else {
                      result[0] += 0.0760891893326321;
                    }
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.637949228286744052) ) ) {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.51693725585937678) ) ) {
                        result[0] += 0.003941904448623992;
                      } else {
                        result[0] += -0.08068696785211685;
                      }
                    } else {
                      if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.861792564392090288) ) ) {
                        if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.673553824424744096) ) ) {
                          result[0] += -0.08749606586207848;
                        } else {
                          result[0] += 0.027532287267227562;
                        }
                      } else {
                        result[0] += -0.12566069729846743;
                      }
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += -0.010826689108169784;
              } else {
                result[0] += -0.05437966137745495;
              }
            }
          } else {
            result[0] += -0.021475375984742334;
          }
        } else {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.249904870986938921) ) ) {
            if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.12219793799430308;
            } else {
              result[0] += -0.0027748489371745453;
            }
          } else {
            result[0] += -0.01392529837838212;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
        if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)10.86439704895019709) ) ) {
          if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.53326439857482999) ) ) {
              result[0] += -0.04882946765618613;
            } else {
              result[0] += -0.12624135276394988;
            }
          } else {
            if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.855921268463135654) ) ) {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)4.331511974334717685) ) ) {
                result[0] += -0.050249615557581576;
              } else {
                result[0] += 0.04144346400065432;
              }
            } else {
              result[0] += 0.0971758031054793;
            }
          }
        } else {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
            result[0] += -0.0005725144028588135;
          } else {
            result[0] += 0.1716580533633262;
          }
        }
      } else {
        result[0] += -5.023051975207532e-05;
      }
    }
  }
  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
    if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)17.27274703979492543) ) ) {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
          if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
            if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += 0.001558527048331263;
            } else {
              if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.15100884437561124) ) ) {
                  if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)7.008503675460816318) ) ) {
                    result[0] += 0.007650060595305506;
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.81821727752685725) ) ) {
                      result[0] += 0.02140509914944385;
                    } else {
                      result[0] += 0.1878100622370088;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.205624103546144354) ) ) {
                    result[0] += 0.09957667694858867;
                  } else {
                    result[0] += 0.02267906207361091;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.09398412704467951) ) ) {
                    result[0] += 0.013950123886020719;
                  } else {
                    result[0] += 0.10467727371084504;
                  }
                } else {
                  if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2727.500000000000455) ) ) {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.18965101242065607) ) ) {
                        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                          result[0] += 0.07139524937822814;
                        } else {
                          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                            result[0] += -0.005785452198877637;
                          } else {
                            if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.01634240150451749) ) ) {
                              result[0] += 0.0005832643870534785;
                            } else {
                              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.623839378356934482) ) ) {
                                result[0] += 0.13933056187202678;
                              } else {
                                result[0] += 2.445790498637712e-05;
                              }
                            }
                          }
                        }
                      } else {
                        result[0] += -0.04459450837818467;
                      }
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
                        result[0] += 0.04046217115780565;
                      } else {
                        result[0] += -0.13283168606868362;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.531673669815064365) ) ) {
                      result[0] += -0.11826723005579276;
                    } else {
                      result[0] += -0.017045979317165342;
                    }
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.41263532638549982) ) ) {
              if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.201251745223999912) ) ) {
                result[0] += -0.00435955507407087;
              } else {
                if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.15100884437561124) ) ) {
                    result[0] += -0.004530260153716831;
                  } else {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.620046615600586826) ) ) {
                      result[0] += -0.2941388897321759;
                    } else {
                      result[0] += 0.006020289604671225;
                    }
                  }
                } else {
                  result[0] += -0.12056724915654947;
                }
              }
            } else {
              result[0] += -0.07910352856940392;
            }
          }
        } else {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                result[0] += -0.025193223160225542;
              } else {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += 0.022996292746175893;
                } else {
                  result[0] += -0.025835461851347954;
                }
              }
            } else {
              result[0] += -0.05726546144152304;
            }
          } else {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.142630577087403232) ) ) {
              result[0] += 0.021119731951506515;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.921924352645874468) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.255632162094117099) ) ) {
                  result[0] += 0.02998662599775131;
                } else {
                  result[0] += -0.018250413862796355;
                }
              } else {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                  result[0] += -0.020530946367390484;
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.450390577316285068) ) ) {
                    result[0] += -0.005162712046702644;
                  } else {
                    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += -0.08707694379763847;
                    } else {
                      result[0] += 0.12455736453637861;
                    }
                  }
                }
              }
            }
          }
        }
      } else {
        result[0] += -0.17622950057400666;
      }
    } else {
      if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
        if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)7.181210994720459873) ) ) {
          result[0] += -0.0057265070071948945;
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.30853915214538663) ) ) {
            result[0] += -0.18908180404597333;
          } else {
            if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)7.223295450210572177) ) ) {
              result[0] += -0.10785937545478108;
            } else {
              if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)7.27828097343444913) ) ) {
                result[0] += 0.03126422120575236;
              } else {
                result[0] += -0.02415402785503227;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
          if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)10.86439704895019709) ) ) {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.242453336715698464) ) ) {
              result[0] += 0.00015320912319371339;
            } else {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.11106884450861415;
              } else {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                  result[0] += -0.06639496356889212;
                } else {
                  result[0] += 0.01157506635080147;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)15.20015096664428889) ) ) {
              result[0] += 0.1560176126889592;
            } else {
              result[0] += -0.019270631916438383;
            }
          }
        } else {
          result[0] += -1.4424700796247507e-05;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.917405366897583452) ) ) {
      if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)2.012675821781158891) ) ) {
        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.623839378356934482) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.605120182037354404) ) ) {
              result[0] += 0.027407713488560356;
            } else {
              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.033102942288869244;
                } else {
                  result[0] += -0.07601981936836466;
                }
              } else {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.04401310775275471;
                } else {
                  result[0] += 0.0031105008483154925;
                }
              }
            }
          } else {
            result[0] += -0.031028242321635337;
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)2.138333082199097124) ) ) {
            if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.003972069221272854;
            } else {
              result[0] += 0.16367948763793883;
            }
          } else {
            if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += -0.061003658035593725;
            } else {
              result[0] += -0.019042727221286904;
            }
          }
        }
      } else {
        result[0] += 0.05352516001303642;
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.439304351806642401) ) ) {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.531673669815064365) ) ) {
          result[0] += 0.042962686258732695;
        } else {
          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)2.249904870986938921) ) ) {
            result[0] += -0.048851497870551853;
          } else {
            result[0] += 0.16411948158390546;
          }
        }
      } else {
        result[0] += -0.08169724662768552;
      }
    }
  }
  if ( UNLIKELY(  (data[30].missing != -1) && (data[30].fvalue <= (double)-1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.138333082199097124) ) ) {
      if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
        result[0] += -0.017505078272472097;
      } else {
        result[0] += -0.1367908237997887;
      }
    } else {
      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.071567356586456743) ) ) {
        if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.051747083663941318) ) ) {
          result[0] += 0.0005826094904698014;
        } else {
          result[0] += -0.19646054632540486;
        }
      } else {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.487163543701172763) ) ) {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.69067406654357999) ) ) {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.441542863845826083) ) ) {
              result[0] += 0.01287839293276302;
            } else {
              result[0] += -0.02033173883580817;
            }
          } else {
            if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.280659198760987216) ) ) {
                  result[0] += 0.05881972545544072;
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
                    if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.861792564392090288) ) ) {
                      result[0] += 0.00647577397857964;
                    } else {
                      result[0] += 0.050733574627542;
                    }
                  } else {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.868834793567657693) ) ) {
                      result[0] += -0.046659094821250115;
                    } else {
                      if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)21466447872.00000381) ) ) {
                        if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.494223117828370029) ) ) {
                          result[0] += 0.0077047149682244066;
                        } else {
                          result[0] += 0.07577978954945025;
                        }
                      } else {
                        result[0] += -0.0628590573873301;
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.11718712562467658;
                } else {
                  if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.53326439857482999) ) ) {
                      result[0] += -0.0023750863364189385;
                    } else {
                      result[0] += 0.07699572975844488;
                    }
                  } else {
                    result[0] += 0.1325769071576459;
                  }
                }
              }
            } else {
              result[0] += 0.11343641527391146;
            }
          }
        } else {
          result[0] += 0.08115855396121978;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.174569487571716753) ) ) {
        if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.085941076278687412) ) ) {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.924581527709961826) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.553712725639343706) ) ) {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                  result[0] += 0.006679038445455792;
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)1.777674019336700661) ) ) {
                    if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.15186696718187145;
                    } else {
                      result[0] += 0.007660906582856095;
                    }
                  } else {
                    result[0] += -0.006436850808052751;
                  }
                }
              } else {
                result[0] += -0.010064212101164077;
              }
            } else {
              if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += -0.003673301080393404;
              } else {
                if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.861792564392090288) ) ) {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.379217386245728427) ) ) {
                    result[0] += 0.005794751181036241;
                  } else {
                    result[0] += -0.020154700227035985;
                  }
                } else {
                  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += 0.012700717185559858;
                      } else {
                        result[0] += -0.033289438476701565;
                      }
                    } else {
                      result[0] += 0.028446161607836543;
                    }
                  } else {
                    result[0] += 0.04160263993788913;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.94957673549652144) ) ) {
              result[0] += -0.00695760717760506;
            } else {
              result[0] += 0.0011655510303079312;
            }
          }
        } else {
          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += 7.76584938310513e-05;
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.142630577087403232) ) ) {
                  result[0] += 0.0036239061873246606;
                } else {
                  result[0] += -0.019940012083549955;
                }
              }
            } else {
              result[0] += 0.007947686503375406;
            }
          } else {
            result[0] += 0.007816380094978703;
          }
        }
      } else {
        result[0] += -0.0069867921963677766;
      }
    } else {
      if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.48298668861389249) ) ) {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.255632162094117099) ) ) {
          if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)0.8958797454833985485) ) ) {
            result[0] += 0.0010582392672881702;
          } else {
            result[0] += -0.006740263344070365;
          }
        } else {
          if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.418317794799805576) ) ) {
            result[0] += -0.0006295963730168306;
          } else {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.329314231872559482) ) ) {
                    result[0] += -0.0014182041936105134;
                  } else {
                    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                      result[0] += 0.004738546294869153;
                    } else {
                      result[0] += -0.03412483013632033;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.088880300521851474) ) ) {
                    result[0] += -0.006130695679024253;
                  } else {
                    if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                      if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += 0.017771213175869858;
                      } else {
                        if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                          result[0] += -0.05824460861684803;
                        } else {
                          result[0] += 0.016013906850336573;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
                        result[0] += -0.0009355902044603551;
                      } else {
                        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.450390577316285068) ) ) {
                          result[0] += 0.0073186843277472;
                        } else {
                          result[0] += 0.037604373560229304;
                        }
                      }
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.102759599685669833) ) ) {
                  if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.552201986312867099) ) ) {
                      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                        result[0] += -0.013643516969130815;
                      } else {
                        result[0] += -0.041986754533190354;
                      }
                    } else {
                      result[0] += -0.057720370904278756;
                    }
                  } else {
                    if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += 0.01022342417920201;
                    } else {
                      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                          result[0] += 0.017580919705845854;
                        } else {
                          result[0] += -0.0225802265587212;
                        }
                      } else {
                        result[0] += -0.06154590754107569;
                      }
                    }
                  }
                } else {
                  result[0] += 0.002720846294488729;
                }
              }
            } else {
              result[0] += -0.04556988124669332;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.142630577087403232) ) ) {
          result[0] += -0.01833263444718554;
        } else {
          result[0] += -0.0039037209054885733;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.597137451171875888) ) ) {
      if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
        result[0] += -0.01261164784097397;
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.914472818374634233) ) ) {
          if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.524927973747253862) ) ) {
            result[0] += 0.04442099795013411;
          } else {
            result[0] += 0.0004244373684899083;
          }
        } else {
          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
            if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
              result[0] += 0.0030799730388069986;
            } else {
              if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)3072.000000000000455) ) ) {
                result[0] += 0.039285490552470125;
              } else {
                result[0] += -0.006528587531214724;
              }
            }
          } else {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.013491562109087774;
                } else {
                  result[0] += -0.030245175961587337;
                }
              } else {
                if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                    result[0] += -0.09658957676653228;
                  } else {
                    if ( UNLIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.13614716412601455;
                    } else {
                      result[0] += -0.0382342474407674;
                    }
                  }
                } else {
                  result[0] += -0.026472827038714914;
                }
              }
            } else {
              result[0] += 0.003239921839221383;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.903337478637697089) ) ) {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += 0.10214884653914824;
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.154959201812744585) ) ) {
                result[0] += 0.055678046957766514;
              } else {
                result[0] += -0.03495022920950133;
              }
            }
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)1.497866153717041238) ) ) {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
                  result[0] += 0.05050667228872076;
                } else {
                  result[0] += -0.11212611394291111;
                }
              } else {
                result[0] += 0.07597910680185593;
              }
            } else {
              if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.02877676039046982;
                } else {
                  if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
                    if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.013770939852631487;
                    } else {
                      result[0] += 0.10177549126144679;
                    }
                  } else {
                    result[0] += 0.001661080723676051;
                  }
                }
              } else {
                result[0] += 0.024877908748180987;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.09398412704467951) ) ) {
            result[0] += 0.0002838145484948466;
          } else {
            if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += -0.013423194097162958;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.78508520126342951) ) ) {
                result[0] += -0.021716631352386953;
              } else {
                result[0] += -0.0516210751053347;
              }
            }
          }
        }
      } else {
        result[0] += 0.0014159277795005346;
      }
    }
  } else {
    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.83122920989990412) ) ) {
      if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.553712725639343706) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.124530076980591708) ) ) {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.511434078216553178) ) ) {
                result[0] += -0.009341333583069687;
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.82428741455078303) ) ) {
                  result[0] += 0.004528844881278307;
                } else {
                  result[0] += -0.006139102127311382;
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.766185760498047763) ) ) {
                if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                      result[0] += -0.01505914786655844;
                    } else {
                      result[0] += -0.049970153663159766;
                    }
                  } else {
                    if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.433569431304932529) ) ) {
                        result[0] += -0.03255169396540121;
                      } else {
                        result[0] += 0.0003253490948283853;
                      }
                    } else {
                      result[0] += 0.01560258038837388;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += 0.006959696588203149;
                  } else {
                    if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                      result[0] += -0.041943019049473965;
                    } else {
                      result[0] += 0.00063067106134852;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += 0.003069381594491522;
                } else {
                  result[0] += -0.004686706923550632;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.737386107444763628) ) ) {
              result[0] += -0.01956992696591904;
            } else {
              result[0] += 0.009231924108501348;
            }
          }
        } else {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.40000796318054288) ) ) {
              result[0] += 0.0009738293346512592;
            } else {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += -0.005849580097930483;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.733271598815919745) ) ) {
                      result[0] += 0.020339605891211893;
                    } else {
                      if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += -0.0033495172277219023;
                      } else {
                        result[0] += 0.021253110528470454;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.214365959167481357) ) ) {
                    result[0] += -0.01964833176286913;
                  } else {
                    result[0] += -0.0019107262672814396;
                  }
                }
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.102759599685669833) ) ) {
                  result[0] += -0.019847273542842714;
                } else {
                  result[0] += 0.005703056852002237;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.58491539955139249) ) ) {
              result[0] += 0.00628260934626691;
            } else {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.868834793567657693) ) ) {
                if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.02046163818492782;
                  } else {
                    result[0] += 0.02034077604094646;
                  }
                } else {
                  if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.06364040495743166;
                  } else {
                    result[0] += -0.018936677353494587;
                  }
                }
              } else {
                result[0] += 0.073428993307517;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
          result[0] += -0.07370774630793013;
        } else {
          result[0] += -0.01243835166936847;
        }
      }
    } else {
      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
        result[0] += -0.0061789966908404655;
      } else {
        result[0] += -0.033517112812744755;
      }
    }
  }
  if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.87502956390381037) ) ) {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
        if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.534971714019776279) ) ) {
            result[0] += 0.003972098260514883;
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.70123767852783381) ) ) {
              result[0] += -0.004518757550298307;
            } else {
              result[0] += -0.10028223977753442;
            }
          }
        } else {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.673553824424744096) ) ) {
              result[0] += -0.10074623318838836;
            } else {
              result[0] += 8.840914585182819e-06;
            }
          } else {
            result[0] += 0.025497027497261893;
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.129040718078614169) ) ) {
          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.715336322784424716) ) ) {
                result[0] += -0.008331381625845895;
              } else {
                result[0] += -0.16783797478697815;
              }
            } else {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.909855604171753818) ) ) {
                result[0] += 0.0097155950573337;
              } else {
                result[0] += -0.042610255944926345;
              }
            }
          } else {
            result[0] += -0.10213170138206201;
          }
        } else {
          if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.868834793567657693) ) ) {
            result[0] += 0.06013392092408574;
          } else {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.770631790161133257) ) ) {
              result[0] += -0.019927512941301632;
            } else {
              if ( UNLIKELY(  (data[27].missing != -1) && (data[27].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                result[0] += 0.020711954280017375;
              } else {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.861792564392090288) ) ) {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.802696108818054643) ) ) {
                      result[0] += 0.07825510778286035;
                    } else {
                      if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += -0.0539542328048264;
                      } else {
                        result[0] += 0.03825949907595418;
                      }
                    }
                  } else {
                    result[0] += -0.04706367494719507;
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.733271598815919745) ) ) {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
                        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                          result[0] += 0.0030940463335898997;
                        } else {
                          result[0] += -0.09033749633336714;
                        }
                      } else {
                        if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2727.500000000000455) ) ) {
                          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.53326439857482999) ) ) {
                            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.802696108818054643) ) ) {
                              result[0] += -0.007885350283800037;
                            } else {
                              if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.511434078216553178) ) ) {
                                result[0] += 0.08723479663018378;
                              } else {
                                result[0] += 0.012043247492815781;
                              }
                            }
                          } else {
                            result[0] += -7.207746354982236e-05;
                          }
                        } else {
                          result[0] += -0.013686978507377752;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                        if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                          if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.868834793567657693) ) ) {
                            result[0] += 0.015207572483625363;
                          } else {
                            result[0] += 0.10265958677586358;
                          }
                        } else {
                          result[0] += -0.10203762150634031;
                        }
                      } else {
                        if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2727.500000000000455) ) ) {
                          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.810334205627442294) ) ) {
                                result[0] += -0.11696746995969556;
                              } else {
                                result[0] += -0.03391643399381506;
                              }
                            } else {
                              result[0] += 0.08016100049302888;
                            }
                          } else {
                            result[0] += -0.10856747988177698;
                          }
                        } else {
                          result[0] += 0.004993637105906529;
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.637949228286744052) ) ) {
                        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.142630577087403232) ) ) {
                          result[0] += 0.034083933159160855;
                        } else {
                          result[0] += -0.001980612188820793;
                        }
                      } else {
                        if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                          result[0] += 0.07181931134778845;
                        } else {
                          if ( UNLIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                            result[0] += -0.03246786588045985;
                          } else {
                            result[0] += 0.07595731810586681;
                          }
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.344550132751465732) ) ) {
                        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)6.218359947204590732) ) ) {
                          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.778982400894165927) ) ) {
                              result[0] += -0.02694873409622325;
                            } else {
                              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
                                if ( UNLIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                                  result[0] += 0.13190475118292508;
                                } else {
                                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                                    result[0] += 0.07236668973274882;
                                  } else {
                                    result[0] += -0.08635791312593169;
                                  }
                                }
                              } else {
                                result[0] += -0.04448667142106488;
                              }
                            }
                          } else {
                            result[0] += -0.04744382834070716;
                          }
                        } else {
                          result[0] += -0.09181996808501576;
                        }
                      } else {
                        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.266057968139650214) ) ) {
                            result[0] += 0.07599914930477719;
                          } else {
                            if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.791641235351563388) ) ) {
                              if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.624251961708069292) ) ) {
                                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                                  result[0] += -0.05813706575004273;
                                } else {
                                  result[0] += -0.16091078577512086;
                                }
                              } else {
                                result[0] += 0.029086456760523308;
                              }
                            } else {
                              result[0] += 0.032826453037374304;
                            }
                          }
                        } else {
                          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.637949228286744052) ) ) {
                            result[0] += 0.08024808603878177;
                          } else {
                            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.861792564392090288) ) ) {
                              result[0] += 0.05521263253370373;
                            } else {
                              result[0] += -0.06423165626371184;
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
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.802696108818054643) ) ) {
        if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.166635274887085849) ) ) {
              result[0] += -0.03168083497875856;
            } else {
              result[0] += 0.09091547121983505;
            }
          } else {
            if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
              result[0] += -0.06415251033717996;
            } else {
              if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.722943305969239169) ) ) {
                result[0] += -0.048249507420021834;
              } else {
                result[0] += 0.03563469276152378;
              }
            }
          }
        } else {
          result[0] += -0.14090832531685302;
        }
      } else {
        if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.418317794799805576) ) ) {
          result[0] += 0.01800345054878953;
        } else {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
            result[0] += 0.11237123250708006;
          } else {
            result[0] += -0.022689679389200093;
          }
        }
      }
    }
  } else {
    result[0] += -0.0002075466256254518;
  }
  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.465643882751465732) ) ) {
      if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
        result[0] += 6.537335307875007e-05;
      } else {
        result[0] += 0.006359756139322082;
      }
    } else {
      if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
        if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
            result[0] += -0.00017215044985425631;
          } else {
            if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.740319490432739702) ) ) {
                result[0] += -0.012184519633389235;
              } else {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += -0.00365422603728853;
                } else {
                  result[0] += 0.041147131649263355;
                }
              }
            } else {
              if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.009846732876567838;
              } else {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.881510615348816362) ) ) {
                  if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                    result[0] += -0.010536235810228205;
                  } else {
                    result[0] += -0.109650481818164;
                  }
                } else {
                  result[0] += 0.008230479198380911;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                result[0] += -0.0023464105925766086;
              } else {
                result[0] += 0.035191079653298186;
              }
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.357691764831543413) ) ) {
                result[0] += -0.00034524963144617893;
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.69067406654357999) ) ) {
                  result[0] += -0.045278518871595404;
                } else {
                  result[0] += -0.022721358001475213;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.347943067550660068) ) ) {
              result[0] += -0.024919923864342206;
            } else {
              result[0] += 0.003237442336069259;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.388237953186036044) ) ) {
          if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)2.138333082199097124) ) ) {
              result[0] += -0.0650810237984987;
            } else {
              result[0] += 0.008532583757639886;
            }
          } else {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.88435244560241788) ) ) {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                    if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                      result[0] += -0.004064693553075445;
                    } else {
                      result[0] += -0.06035313507622408;
                    }
                  } else {
                    if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += -0.0528588176632642;
                    } else {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.431901693344116655) ) ) {
                        result[0] += 0.039551785946815576;
                      } else {
                        result[0] += -0.015519757311477334;
                      }
                    }
                  }
                } else {
                  result[0] += -0.05057157729345681;
                }
              } else {
                if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += -0.003449974197722541;
                  } else {
                    if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += 0.0810233163275001;
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.935661792755128729) ) ) {
                        result[0] += -0.0020631286955068236;
                      } else {
                        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.13002538681030451) ) ) {
                            result[0] += 0.04017602689421063;
                          } else {
                            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.13022470474243342) ) ) {
                              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.665476083755494052) ) ) {
                                result[0] += -0.0953505139747826;
                              } else {
                                result[0] += 0.0522655637136219;
                              }
                            } else {
                              result[0] += -0.004006054635320159;
                            }
                          }
                        } else {
                          result[0] += 0.03454787586323004;
                        }
                      }
                    }
                  }
                } else {
                  result[0] += -0.005002705672711521;
                }
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
                result[0] += -0.002183025181622054;
              } else {
                if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.029068946838379794) ) ) {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += 0.042783125426400226;
                    } else {
                      result[0] += -0.05472005638839357;
                    }
                  } else {
                    result[0] += 0.003107155355990152;
                  }
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.126885652542115146) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.823630809783937323) ) ) {
                      result[0] += -0.025274868628507516;
                    } else {
                      if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)3.921924352645874468) ) ) {
                        result[0] += -0.052066740030737316;
                      } else {
                        result[0] += 0.03571145838385595;
                      }
                    }
                  } else {
                    result[0] += 0.06180667127023788;
                  }
                }
              }
            }
          }
        } else {
          result[0] += 0.011985224082454263;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.83122920989990412) ) ) {
      if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
        if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
          result[0] += 0.0015058748551285265;
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.940167903900147373) ) ) {
            if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += 0.04247623490517753;
            } else {
              result[0] += -0.044415793125653094;
            }
          } else {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.53326439857482999) ) ) {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += 0.04786887636332155;
              } else {
                result[0] += -0.029704247126062584;
              }
            } else {
              result[0] += 0.06311102343928297;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.553712725639343706) ) ) {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.497866153717041238) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.102759599685669833) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.01588991359588073;
                } else {
                  result[0] += -0.036873460721928875;
                }
              } else {
                result[0] += 0.0247105428038996;
              }
            } else {
              result[0] += 0.07369247255906551;
            }
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.803987503051758701) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.403187274932863104) ) ) {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += -0.0063616380118104365;
                } else {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += 0.015465934137215684;
                  } else {
                    result[0] += -0.0042205237209174224;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.0016586678999827604;
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.467917680740357333) ) ) {
                    result[0] += -0.01321128669389777;
                  } else {
                    result[0] += -0.03806732187897773;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.617236852645874912) ) ) {
                result[0] += 2.302346097688747e-05;
              } else {
                result[0] += -0.008351858398771561;
              }
            }
          }
        } else {
          result[0] += -0.025178257141595686;
        }
      }
    } else {
      result[0] += -0.008093856159879455;
    }
  }
  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.465643882751465732) ) ) {
      result[0] += 0.003071269466211376;
    } else {
      if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
          if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += -0.09846939075292213;
          } else {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += 0.005640224467681143;
            } else {
              result[0] += -0.04115741912482947;
            }
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.433569431304932529) ) ) {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += -0.003266408848205442;
            } else {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.918272972106934482) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.219399690628052646) ) ) {
                  result[0] += -0.04763803481910514;
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.60200452804565607) ) ) {
                    if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
                      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.055311203002930576) ) ) {
                          result[0] += -0.07049093011579018;
                        } else {
                          if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.673553824424744096) ) ) {
                              result[0] += -0.02379879275447931;
                            } else {
                              result[0] += 0.05268001238205194;
                            }
                          } else {
                            result[0] += -0.0025482152593210893;
                          }
                        }
                      } else {
                        result[0] += -0.032517749046736814;
                      }
                    } else {
                      result[0] += -0.07849018691423117;
                    }
                  } else {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
                      result[0] += 0.06997115079597531;
                    } else {
                      result[0] += -0.025555815043977605;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.772694945335388628) ) ) {
                  result[0] += -0.026982448080229267;
                } else {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
                    result[0] += -0.11267963263374328;
                  } else {
                    result[0] += 0.0013627778866227916;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.30853915214538663) ) ) {
              if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                result[0] += 0.004486660405353174;
              } else {
                result[0] += -0.007528670107823686;
              }
            } else {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.909855604171753818) ) ) {
                result[0] += 0.0036337550586370533;
              } else {
                if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += 0.001961002175018849;
                } else {
                  if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.436733961105347568) ) ) {
                    result[0] += -0.002661702353526892;
                  } else {
                    result[0] += -0.02309320355640812;
                  }
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.740319490432739702) ) ) {
          result[0] += -0.0011154733230198293;
        } else {
          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.940167903900147373) ) ) {
              result[0] += 0.09938203546848469;
            } else {
              result[0] += -0.011858287089169592;
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.531673669815064365) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.63218307495117365) ) ) {
                result[0] += -0.011006228613669993;
              } else {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += 0.014384737453953517;
                } else {
                  result[0] += -0.019247767975580928;
                }
              }
            } else {
              result[0] += 0.005656186467606667;
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.991406440734865058) ) ) {
      result[0] += 2.0835025109633624e-05;
    } else {
      if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
        if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
          if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)3.000000000000000444) ) ) {
            result[0] += 0.04804370111971515;
          } else {
            result[0] += -0.006567474059844121;
          }
        } else {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
            result[0] += -0.02325611637059743;
          } else {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.431901693344116655) ) ) {
              result[0] += -0.01009284557348459;
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.04641723632812678) ) ) {
                  result[0] += -0.12888060153516384;
                } else {
                  result[0] += 0.04797124244713898;
                }
              } else {
                if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.90474271774292081) ) ) {
                    result[0] += 0.008798124397761644;
                  } else {
                    result[0] += -0.017179294426798337;
                  }
                } else {
                  result[0] += -0.04075120429750364;
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.28299736976623624) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.637949228286744052) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += 0.007515063164403414;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.094205617904663974) ) ) {
                  result[0] += -0.07286992735496889;
                } else {
                  result[0] += -0.011006760548696093;
                }
              }
            } else {
              result[0] += -0.05162196039845718;
            }
          } else {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += -0.0016234297195451158;
            } else {
              if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.003240342146016022;
                } else {
                  result[0] += 0.04600506062051258;
                }
              } else {
                result[0] += 0.056200564521782986;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
            if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.497866153717041238) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.803987503051758701) ) ) {
                if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.007347171113092167;
                } else {
                  result[0] += -0.03422247889914403;
                }
              } else {
                if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.384246587753296343) ) ) {
                      result[0] += -0.0021567869371270916;
                    } else {
                      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                        result[0] += -0.006915259549898441;
                      } else {
                        result[0] += -0.03397539483049728;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.05831733855086359;
                    } else {
                      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                        result[0] += -0.033395320655126376;
                      } else {
                        result[0] += 0.013656778627429431;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += 0.001286720317854296;
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
                      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
                        result[0] += 0.029259156489610416;
                      } else {
                        result[0] += -0.044349955453717144;
                      }
                    } else {
                      result[0] += -0.013557670296050726;
                    }
                  }
                }
              }
            } else {
              result[0] += 0.011166742118490077;
            }
          } else {
            if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += -0.009483483216454851;
            } else {
              result[0] += -0.03984005127222357;
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.000000000000000444) ) ) {
    if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.69067406654357999) ) ) {
        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
          result[0] += 0.09848366641027925;
        } else {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.994480729103088823) ) ) {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.636499762535095659) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.510617971420288974) ) ) {
                result[0] += 0.04709501339971284;
              } else {
                result[0] += -0.009537046055003459;
              }
            } else {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.247576236724854404) ) ) {
                result[0] += -0.10959364386667264;
              } else {
                result[0] += -0.005735215123218693;
              }
            }
          } else {
            if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += 0.08730003869420153;
              } else {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += 0.046800046361360995;
                } else {
                  result[0] += -0.042262995635422906;
                }
              }
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.060294389724732333) ) ) {
                result[0] += -0.07471650230209953;
              } else {
                result[0] += 0.039405349857437225;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.802696108818054643) ) ) {
          result[0] += -0.14636650039986882;
        } else {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.002088938847295772;
            } else {
              result[0] += -0.119658310266549;
            }
          } else {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.249904870986938921) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.334978580474854404) ) ) {
                  if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.673553824424744096) ) ) {
                    result[0] += -0.009154240868424195;
                  } else {
                    result[0] += 0.06440055157701113;
                  }
                } else {
                  result[0] += -0.05230413045537679;
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.637949228286744052) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.53326439857482999) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.18088722229004084) ) ) {
                      result[0] += -0.16223276048253799;
                    } else {
                      result[0] += -0.026558745305385552;
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.733271598815919745) ) ) {
                      result[0] += 0.11142405779412441;
                    } else {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.941534638404846635) ) ) {
                        result[0] += -0.11778071574177447;
                      } else {
                        result[0] += 0.019079307327148687;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.158952236175537998) ) ) {
                      result[0] += 0.07157608834852955;
                    } else {
                      result[0] += -0.03175093801526119;
                    }
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.450390577316285068) ) ) {
                      result[0] += -0.028696276376087522;
                    } else {
                      result[0] += 0.05946263646540424;
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.141444921493531162) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.173939466476441318) ) ) {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += 0.0156432162708036;
                  } else {
                    result[0] += -0.07336997197503718;
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.464467763900757724) ) ) {
                    result[0] += -0.13135164875553063;
                  } else {
                    result[0] += -0.008203872435179794;
                  }
                }
              } else {
                result[0] += -0.06891967676842188;
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
        result[0] += -0.0013350555614882063;
      } else {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.617236852645874912) ) ) {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.700598716735840066) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.025192260742188388) ) ) {
              result[0] += 0.1056597049535944;
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.605039834976196733) ) ) {
                result[0] += -0.04201726992479477;
              } else {
                result[0] += 0.04559014677979609;
              }
            }
          } else {
            if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1.868834793567657693) ) ) {
              result[0] += 0.09832134681594498;
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.511434078216553178) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.262283086776734287) ) ) {
                  result[0] += 0.0608026179656918;
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
                    result[0] += -0.1442567707620704;
                  } else {
                    if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)9.884762287139894354) ) ) {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.35311269760132014) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.664408206939698154) ) ) {
                          result[0] += 0.07489170377972149;
                        } else {
                          result[0] += -0.011709885571166973;
                        }
                      } else {
                        result[0] += 0.06405205175981098;
                      }
                    } else {
                      result[0] += -0.10488019561140649;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.938867926597595659) ) ) {
                  result[0] += -0.1071768874548143;
                } else {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.342454433441162998) ) ) {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.436733961105347568) ) ) {
                        result[0] += -0.10320056843455135;
                      } else {
                        result[0] += 0.020045850808752756;
                      }
                    } else {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.153024196624756748) ) ) {
                        if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += 0.14049640030945074;
                        } else {
                          result[0] += 0.006863976368006998;
                        }
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.637949228286744052) ) ) {
                          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.791641235351563388) ) ) {
                            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.921060562133789951) ) ) {
                              result[0] += 0.10907914535305313;
                            } else {
                              result[0] += -0.013405077478702935;
                            }
                          } else {
                            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.51517200469970881) ) ) {
                              result[0] += -0.09428548960316176;
                            } else {
                              result[0] += 0.020391743630319713;
                            }
                          }
                        } else {
                          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.135134458541871005) ) ) {
                            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.216319084167481357) ) ) {
                              result[0] += -0.18211436594679797;
                            } else {
                              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.941534638404846635) ) ) {
                                result[0] += 0.09815802738601224;
                              } else {
                                result[0] += -0.01907149667217029;
                              }
                            }
                          } else {
                            result[0] += 0.09447966903717982;
                          }
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.676220536231995073) ) ) {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.921924352645874468) ) ) {
                        result[0] += 0.11496466374821973;
                      } else {
                        result[0] += 0.02084996721144068;
                      }
                    } else {
                      result[0] += -0.01864855713530076;
                    }
                  }
                }
              }
            }
          }
        } else {
          result[0] += -0.0733613877190498;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)6.000000000000000888) ) ) {
      if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.029068946838379794) ) ) {
          result[0] += 0.00740338649674267;
        } else {
          result[0] += -0.03797762094585129;
        }
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.105651378631592685) ) ) {
          result[0] += -0.022576377766372602;
        } else {
          result[0] += 0.038959041499283316;
        }
      }
    } else {
      result[0] += -0.00011695228290816624;
    }
  }
  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
    result[0] += -0.000521055139381067;
  } else {
    if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.740319490432739702) ) ) {
      result[0] += -0.0004797770202958397;
    } else {
      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
        if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.41263532638549982) ) ) {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.397998809814454013) ) ) {
                result[0] += 0.009912977298980677;
              } else {
                result[0] += -0.06771099830345999;
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.397998809814454013) ) ) {
                if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.046640634536744052) ) ) {
                  result[0] += -0.25172875113503984;
                } else {
                  if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.540854334831238237) ) ) {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.241523027420044833) ) ) {
                          result[0] += -0.00020614020439603444;
                        } else {
                          result[0] += 0.041209984427793674;
                        }
                      } else {
                        result[0] += 0.044563491954936335;
                      }
                    } else {
                      if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.238486170768738237) ) ) {
                        result[0] += 0.03283999632450391;
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
                          if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)11.09085798263549982) ) ) {
                            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.680161952972413886) ) ) {
                              if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.593729019165039951) ) ) {
                                result[0] += -0.08311386914769048;
                              } else {
                                result[0] += 0.0059519406785241355;
                              }
                            } else {
                              result[0] += -0.07018643570684094;
                            }
                          } else {
                            result[0] += 0.13062425027623156;
                          }
                        } else {
                          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.131699204444885698) ) ) {
                            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.13002538681030451) ) ) {
                              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                                result[0] += -0.17972908350605535;
                              } else {
                                result[0] += -0.03166251878810073;
                              }
                            } else {
                              result[0] += 0.015390033336184604;
                            }
                          } else {
                            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.431901693344116655) ) ) {
                              result[0] += 0.0926167105715226;
                            } else {
                              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.851041555404663974) ) ) {
                                result[0] += -0.005454348941775507;
                              } else {
                                result[0] += -0.1312751675970754;
                              }
                            }
                          }
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                      result[0] += 0.04190943690760619;
                    } else {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.700598716735840066) ) ) {
                        result[0] += -0.002654767206936172;
                      } else {
                        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                          result[0] += 0.015554398025257555;
                        } else {
                          result[0] += -0.16395179456275671;
                        }
                      }
                    }
                  }
                }
              } else {
                result[0] += 0.011479008826853362;
              }
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
              result[0] += 0.029279113869504332;
            } else {
              result[0] += -0.027216230620071426;
            }
          }
        } else {
          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += -0.02194377056181676;
            } else {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.197173833847046787) ) ) {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.650573849678039995) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.54220247268676935) ) ) {
                      if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)10.89387273788452326) ) ) {
                        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.868834793567657693) ) ) {
                          result[0] += 0.003292736899056354;
                        } else {
                          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.344550132751465732) ) ) {
                            result[0] += 0.10393991475903573;
                          } else {
                            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.041921615600587714) ) ) {
                              result[0] += 0.10017473615162287;
                            } else {
                              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.991406440734865058) ) ) {
                                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.643222332000734198) ) ) {
                                  result[0] += -0.03824374593115482;
                                } else {
                                  result[0] += -0.23627789062508528;
                                }
                              } else {
                                result[0] += 0.09323189492200382;
                              }
                            }
                          }
                        }
                      } else {
                        result[0] += -0.3313685511781548;
                      }
                    } else {
                      result[0] += 0.12501196435160603;
                    }
                  } else {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.921060562133789951) ) ) {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.14095449447632014) ) ) {
                        result[0] += 0.058889694721099596;
                      } else {
                        result[0] += -0.18876656747196224;
                      }
                    } else {
                      result[0] += -0.2707900001117544;
                    }
                  }
                } else {
                  result[0] += 0.10483141709190186;
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.898905277252199042) ) ) {
                  result[0] += 0.017947427252554413;
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.025192260742188388) ) ) {
                    result[0] += 0.0222554638908369;
                  } else {
                    result[0] += -0.04990952825577577;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.725620865821838823) ) ) {
              result[0] += -0.0008034735362325209;
            } else {
              result[0] += 0.006404919773171153;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.129040718078614169) ) ) {
          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += -0.1716984392143522;
            } else {
              result[0] += 0.027288369062418067;
            }
          } else {
            if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += 0.0813050316885587;
            } else {
              result[0] += 0.008148070920287116;
            }
          }
        } else {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.715336322784424716) ) ) {
            if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += -0.0050627526411391385;
            } else {
              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.740319490432739702) ) ) {
                result[0] += 0.046473374376261235;
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.154959201812744585) ) ) {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += 0.01876699394892856;
                  } else {
                    result[0] += -0.11894831551932197;
                  }
                } else {
                  result[0] += 0.0027459282958469007;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.166635274887085849) ) ) {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.350240230560303178) ) ) {
                    result[0] += 0.0003453651224499832;
                  } else {
                    result[0] += 0.10144040343339178;
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.81940793991089045) ) ) {
                    result[0] += -0.1577541791864854;
                  } else {
                    result[0] += 0.03742763562704044;
                  }
                }
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.601370334625245029) ) ) {
                  result[0] += -0.027027033133605216;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.474771499633789951) ) ) {
                    result[0] += -0.040678237223067494;
                  } else {
                    result[0] += -0.14888493885545803;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.654679536819458896) ) ) {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += -0.017518801085124328;
                } else {
                  result[0] += 0.09367307340729879;
                }
              } else {
                result[0] += -0.023765600298737706;
              }
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.737386107444763628) ) ) {
      if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)12.00000000000000178) ) ) {
        result[0] += 0.0062019022332032575;
      } else {
        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
            result[0] += -0.017112269212515605;
          } else {
            result[0] += 0.06915866653085576;
          }
        } else {
          if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.673553824424744096) ) ) {
            result[0] += 0.024925246309802543;
          } else {
            result[0] += -0.04435295021442473;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
        if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)3.000000000000000444) ) ) {
          result[0] += -0.006681112003917446;
        } else {
          if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.12316194910306039;
            } else {
              if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.302512168884278232) ) ) {
                  if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                    result[0] += 0.0157101846959202;
                  } else {
                    result[0] += -0.01840708837521478;
                  }
                } else {
                  result[0] += -0.006589124781495028;
                }
              } else {
                if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)11.62723302841186701) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.247576236724854404) ) ) {
                    result[0] += -0.02455152574320565;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.766185760498047763) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.509355545043946201) ) ) {
                        result[0] += -0.06937896692635458;
                      } else {
                        result[0] += 0.0055038502017331166;
                      }
                    } else {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.80468511581421076) ) ) {
                        result[0] += 0.03596592492814715;
                      } else {
                        if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.285887241363526279) ) ) {
                            result[0] += 0.011751644244933912;
                          } else {
                            result[0] += -0.06274143921938578;
                          }
                        } else {
                          result[0] += 0.04910697289547855;
                        }
                      }
                    }
                  }
                } else {
                  result[0] += 0.1693939358751857;
                }
              }
            }
          } else {
            result[0] += -0.00027393750282956366;
          }
        }
      } else {
        if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.242453336715698464) ) ) {
            result[0] += 0.16239791273101056;
          } else {
            result[0] += 0.00023345381209898962;
          }
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.329314231872559482) ) ) {
            if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.001396832406653819;
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.216319084167481357) ) ) {
                result[0] += 0.03254674405964072;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.158509254455567294) ) ) {
                  result[0] += -0.11602506294131627;
                } else {
                  result[0] += 0.013538263872739582;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
              result[0] += 0.028818329476952804;
            } else {
              result[0] += -0.06657528181608599;
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.967588424682618964) ) ) {
      if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.24173307418823331) ) ) {
          result[0] += 0.007664593094157306;
        } else {
          if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += -0.020149969914276297;
          } else {
            result[0] += -0.0010638857614434266;
          }
        }
      } else {
        result[0] += 0.0002866433971145692;
      }
    } else {
      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
        if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
            if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.321723937988282138) ) ) {
                result[0] += -0.0009743429428328483;
              } else {
                result[0] += -0.04848812041330364;
              }
            } else {
              result[0] += 0.09787034870506336;
            }
          } else {
            if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
              if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += -0.00021210554303772;
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.344550132751465732) ) ) {
                  result[0] += -0.00737715594043808;
                } else {
                  result[0] += -0.06253279035991498;
                }
              }
            } else {
              if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                  result[0] += 0.031378728805350614;
                } else {
                  if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.497866153717041238) ) ) {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.384830474853516513) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.431901693344116655) ) ) {
                        result[0] += 0.024742028395406745;
                      } else {
                        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                            result[0] += -0.057203346716677345;
                          } else {
                            if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.53326439857482999) ) ) {
                                result[0] += -0.033129410101461705;
                              } else {
                                result[0] += 0.022096758761135194;
                              }
                            } else {
                              result[0] += -0.06380951797178525;
                            }
                          }
                        } else {
                          result[0] += -0.08300402401543851;
                        }
                      }
                    } else {
                      result[0] += -0.0017339131132172942;
                    }
                  } else {
                    result[0] += 0.0007892877704159468;
                  }
                }
              } else {
                result[0] += 0.010021086163554566;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
            result[0] += -0.012304168525902574;
          } else {
            if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.947818994522095615) ) ) {
                result[0] += 0.002846668812388766;
              } else {
                result[0] += -0.01970443777313092;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.12508964538574396) ) ) {
                result[0] += -0.04873515698154467;
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.780892848968506748) ) ) {
                  result[0] += -0.008562900230608834;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.35311269760132014) ) ) {
                    if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.349750161170959917) ) ) {
                      result[0] += -0.14027260258419078;
                    } else {
                      result[0] += 0.04967130075600379;
                    }
                  } else {
                    if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.349750161170959917) ) ) {
                      result[0] += 0.12666905783631585;
                    } else {
                      if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += -0.03968609232075633;
                      } else {
                        result[0] += 0.12766526679025036;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.497866153717041238) ) ) {
          if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.03420138359069913) ) ) {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.637949228286744052) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.09806728363037287) ) ) {
                result[0] += 0.05526658832118197;
              } else {
                result[0] += -0.02939231352893966;
              }
            } else {
              result[0] += 0.014181885896957628;
            }
          } else {
            result[0] += -0.0347684625350647;
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.510617971420288974) ) ) {
            result[0] += -0.012526429976802728;
          } else {
            result[0] += 0.03302568112332745;
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.000000000000000444) ) ) {
    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.105651378631592685) ) ) {
      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.51389026641845881) ) ) {
        if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.097527027130127841) ) ) {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.947818994522095615) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.737386107444763628) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.431901693344116655) ) ) {
                  result[0] += 0.08054891261423319;
                } else {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)6.218359947204590732) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.43267917633056818) ) ) {
                      result[0] += -0.012022729772513006;
                    } else {
                      result[0] += 0.033532681907237176;
                    }
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.778982400894165927) ) ) {
                      result[0] += -0.11775118984087957;
                    } else {
                      result[0] += -0.01571419906433198;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.709793567657472479) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.542080402374269354) ) ) {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.012675821781158891) ) ) {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.825422286987305576) ) ) {
                        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.347943067550660068) ) ) {
                          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.241523027420044833) ) ) {
                            result[0] += -0.010355312699703509;
                          } else {
                            result[0] += 0.09408729673821672;
                          }
                        } else {
                          if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.569433569908142534) ) ) {
                            result[0] += -0.04060918245772051;
                          } else {
                            result[0] += 0.05571175593704472;
                          }
                        }
                      } else {
                        result[0] += -0.04079887294403279;
                      }
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.242078304290772373) ) ) {
                        result[0] += -0.0341772923129026;
                      } else {
                        result[0] += 0.01554775081072547;
                      }
                    }
                  } else {
                    result[0] += -0.09025932324707545;
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.815814018249513495) ) ) {
                    result[0] += 0.08836146614873625;
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.802696108818054643) ) ) {
                      result[0] += -0.006500611986961467;
                    } else {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.44381141662597834) ) ) {
                        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.51918649673462092) ) ) {
                          result[0] += 0.032490524995847425;
                        } else {
                          result[0] += 0.08776054119203518;
                        }
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.060294389724732333) ) ) {
                          result[0] += 0.09325247193762401;
                        } else {
                          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.342454433441162998) ) ) {
                            result[0] += -0.03255200019828235;
                          } else {
                            result[0] += 0.028789911131716573;
                          }
                        }
                      }
                    }
                  }
                }
              }
            } else {
              result[0] += 0.13371192801459575;
            }
          } else {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.35306882858276456) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.24173307418823331) ) ) {
                result[0] += -0.07597348862138581;
              } else {
                result[0] += -0.011408640618291721;
              }
            } else {
              if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += -0.026657534682378775;
              } else {
                result[0] += 0.10953452425016588;
              }
            }
          }
        } else {
          result[0] += -0.11255832098578995;
        }
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.94957673549652144) ) ) {
          result[0] += 0.026284103886765133;
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.431901693344116655) ) ) {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
              result[0] += -0.16982644134871358;
            } else {
              result[0] += -0.009222223561181287;
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.510617971420288974) ) ) {
              result[0] += 0.02152291678055633;
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)14.19447278976440607) ) ) {
                result[0] += -0.07012488990232214;
              } else {
                result[0] += 0.017234704378816423;
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
        if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
          if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.068990230560303623) ) ) {
            result[0] += -0.005618993007999973;
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.909855604171753818) ) ) {
              if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.31402075290679976) ) ) {
                result[0] += 0.01897621232484659;
              } else {
                if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.007160314033784482;
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.53326439857482999) ) ) {
                    result[0] += 0.01855539737635883;
                  } else {
                    result[0] += -0.15565524973090256;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.481121778488159624) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.53326439857482999) ) ) {
                  result[0] += 0.0023925872156794164;
                } else {
                  if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.06530847847191518;
                  } else {
                    result[0] += 0.17419231549944092;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.695914268493653232) ) ) {
                  if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.48298668861389249) ) ) {
                    result[0] += -0.004000745365671005;
                  } else {
                    result[0] += -0.09958869689566235;
                  }
                } else {
                  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.004910983191482054;
                  } else {
                    result[0] += 0.07587525153472559;
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.24173307418823331) ) ) {
            result[0] += 0.0479922087313502;
          } else {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.628996372222901279) ) ) {
              if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)6.58686089515686124) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.53326439857482999) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.778982400894165927) ) ) {
                    result[0] += 0.013654514331626464;
                  } else {
                    result[0] += -0.06993173439180031;
                  }
                } else {
                  result[0] += -0.007973573264762734;
                }
              } else {
                result[0] += -0.08557509628312981;
              }
            } else {
              result[0] += -0.10625500699722588;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.03420138359069913) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.628555774688722479) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.318498134613038886) ) ) {
              if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += 0.018501385272429404;
                } else {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.23832273483276456) ) ) {
                    result[0] += -0.0218593719901633;
                  } else {
                    result[0] += -0.13206113720020715;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.921060562133789951) ) ) {
                  result[0] += 0.09295784303668395;
                } else {
                  result[0] += -0.0019131365562482503;
                }
              }
            } else {
              result[0] += 0.08374780185214134;
            }
          } else {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.020127415657043901) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.637949228286744052) ) ) {
                result[0] += -0.09862286482064889;
              } else {
                result[0] += 0.02509795930303235;
              }
            } else {
              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.673553824424744096) ) ) {
                result[0] += -0.037135855200418726;
              } else {
                result[0] += 0.01096418608436193;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
            result[0] += 0.15528529185076667;
          } else {
            result[0] += 0.009756292454861558;
          }
        }
      }
    }
  } else {
    result[0] += -0.0003352368265465093;
  }
  if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)8.071062088012697089) ) ) {
    if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.000000000000000444) ) ) {
      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.153024196624756748) ) ) {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.51389026641845881) ) ) {
          if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.03420138359069913) ) ) {
            if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.998158693313599077) ) ) {
              if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.650573849678039995) ) ) {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.855921268463135654) ) ) {
                    result[0] += 0.01002191535362838;
                  } else {
                    if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.02680390151365259;
                    } else {
                      result[0] += 0.025813867109344215;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.643222332000734198) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.088880300521851474) ) ) {
                      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.481121778488159624) ) ) {
                          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.53326439857482999) ) ) {
                            result[0] += 0.047722444142913586;
                          } else {
                            result[0] += -0.06891547630968835;
                          }
                        } else {
                          result[0] += -0.13149980250113166;
                        }
                      } else {
                        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.198464870452881303) ) ) {
                          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.778982400894165927) ) ) {
                            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.467917680740357333) ) ) {
                              result[0] += -0.00889114061050748;
                            } else {
                              result[0] += 0.0930368634544125;
                            }
                          } else {
                            result[0] += -0.06202400011538868;
                          }
                        } else {
                          result[0] += 0.06372425355333403;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.00396752357482999) ) ) {
                        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                          result[0] += 0.06973706303689474;
                        } else {
                          result[0] += 0.01608048758859957;
                        }
                      } else {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.35306882858276456) ) ) {
                          if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                            if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                              result[0] += -0.018396800199936707;
                            } else {
                              result[0] += -0.12260039277427702;
                            }
                          } else {
                            result[0] += 0.011947143781696978;
                          }
                        } else {
                          result[0] += 0.04382135305577449;
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                        if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += 0.13207207562475215;
                        } else {
                          result[0] += 0.04769405859835125;
                        }
                      } else {
                        result[0] += 0.019333063281634238;
                      }
                    } else {
                      result[0] += 0.017299914081243623;
                    }
                  }
                }
              } else {
                result[0] += 0.08402840405114892;
              }
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.53326439857482999) ) ) {
                result[0] += -0.01453614349534793;
              } else {
                result[0] += -0.12164489978944569;
              }
            }
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)1.777674019336700661) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)1.497866153717041238) ) ) {
                result[0] += -0.010478417647220482;
              } else {
                result[0] += -0.1975032872988174;
              }
            } else {
              if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.434600353240968573) ) ) {
                  result[0] += 0.07886399361216556;
                } else {
                  result[0] += 0.023313365111463556;
                }
              } else {
                result[0] += 0.18226036722966021;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.94957673549652144) ) ) {
            result[0] += 0.01901697631068169;
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.431901693344116655) ) ) {
              if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                result[0] += -0.1570118656943073;
              } else {
                result[0] += 0.013922906047930084;
              }
            } else {
              result[0] += -0.02399204334859216;
            }
          }
        }
      } else {
        result[0] += -0.0006611452560004466;
      }
    } else {
      if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)6.000000000000000888) ) ) {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.35306882858276456) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.303166389465332919) ) ) {
            if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
              if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.13022470474243342) ) ) {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.357691764831543413) ) ) {
                    result[0] += 0.049246676786698806;
                  } else {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                      result[0] += -0.05118045291613565;
                    } else {
                      result[0] += -0.10998611030727315;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += -0.05380856936409842;
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.60838890075683771) ) ) {
                      if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.349750161170959917) ) ) {
                        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.135134458541871005) ) ) {
                          result[0] += 0.09830629636611811;
                        } else {
                          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.400584220886231357) ) ) {
                            result[0] += -0.10648431224272692;
                          } else {
                            result[0] += 0.04716752885470507;
                          }
                        }
                      } else {
                        result[0] += -0.09065958097306043;
                      }
                    } else {
                      result[0] += 0.07372338896821011;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.01634240150451749) ) ) {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.417592287063599077) ) ) {
                    result[0] += 0.011642157696727615;
                  } else {
                    result[0] += 0.09900254635914157;
                  }
                } else {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.863673448562622958) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.12789058685302912) ) ) {
                      result[0] += 0.002144193062530625;
                    } else {
                      result[0] += -0.12341140227172159;
                    }
                  } else {
                    if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.602003335952759233) ) ) {
                      result[0] += -0.0734694345996184;
                    } else {
                      result[0] += 0.11645272914169165;
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.766185760498047763) ) ) {
                result[0] += -0.047430651987429844;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.262283086776734287) ) ) {
                  result[0] += -0.06510124254348985;
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.255632162094117099) ) ) {
                    if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.733271598815919745) ) ) {
                        result[0] += 0.05256272516805777;
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.266057968139650214) ) ) {
                          result[0] += -0.09570782031231002;
                        } else {
                          if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.01634240150451749) ) ) {
                            result[0] += -0.02456130571602355;
                          } else {
                            result[0] += 0.07171432405376003;
                          }
                        }
                      }
                    } else {
                      result[0] += 0.0675043740839151;
                    }
                  } else {
                    result[0] += 0.05886686156963405;
                  }
                }
              }
            }
          } else {
            result[0] += -0.14794204163236924;
          }
        } else {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.434600353240968573) ) ) {
            if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.673553824424744096) ) ) {
              result[0] += -0.011966702129435127;
            } else {
              result[0] += 0.09215250498922273;
            }
          } else {
            result[0] += 0.1231864041275018;
          }
        }
      } else {
        result[0] += -0.0002715402891000106;
      }
    }
  } else {
    if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)8.148167133331300604) ) ) {
      result[0] += -0.11130435483465038;
    } else {
      result[0] += 0.001577746698395622;
    }
  }
  if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)8.071062088012697089) ) ) {
    if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
      if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
        result[0] += -0.012230574215186519;
      } else {
        if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.179772853851319248) ) ) {
            result[0] += 0.0009106252986915177;
          } else {
            result[0] += -0.0069559025298418994;
          }
        } else {
          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
            if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.909855604171753818) ) ) {
                result[0] += 0.0005661300806265768;
              } else {
                result[0] += -0.01093584047679299;
              }
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.43749904632568537) ) ) {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                  result[0] += -0.01997580218228543;
                } else {
                  result[0] += 0.045346986304437946;
                }
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.53326439857482999) ) ) {
                  result[0] += -0.0022451310264115566;
                } else {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += 0.020997791659883566;
                  } else {
                    result[0] += 0.10078809223294574;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
              if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)3.000000000000000444) ) ) {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.357691764831543413) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.55753517150879084) ) ) {
                      result[0] += -0.021027766353174077;
                    } else {
                      result[0] += -0.05693281191310014;
                    }
                  } else {
                    result[0] += 0.046682179301541296;
                  }
                } else {
                  result[0] += -0.002925138834325437;
                }
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += 0.06466968752825643;
                } else {
                  result[0] += 0.00443253653162297;
                }
              }
            } else {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
                if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)8.285748958587648261) ) ) {
                    if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.553712725639343706) ) ) {
                        if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)6.796264648437500888) ) ) {
                          result[0] += -0.003614113404142499;
                        } else {
                          result[0] += 0.09094656716923044;
                        }
                      } else {
                        result[0] += -0.03348839534262196;
                      }
                    } else {
                      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                          result[0] += -0.03385309181653348;
                        } else {
                          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.531673669815064365) ) ) {
                            result[0] += -0.05440604088147928;
                          } else {
                            result[0] += 0.011490937345968927;
                          }
                        }
                      } else {
                        result[0] += -0.0024023952154490057;
                      }
                    }
                  } else {
                    result[0] += 0.12294338280199966;
                  }
                } else {
                  if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += -0.00471776822631198;
                  } else {
                    if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.357691764831543413) ) ) {
                      result[0] += -0.06398805435406724;
                    } else {
                      result[0] += 0.03472049072483405;
                    }
                  }
                }
              } else {
                result[0] += -0.050789453627236705;
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.384246587753296343) ) ) {
        if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.384246587753296343) ) ) {
          result[0] += -0.0007622692627520886;
        } else {
          result[0] += 0.012972321675210247;
        }
      } else {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
          if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.918693304061890537) ) ) {
              result[0] += 0.019318711844674544;
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.321723937988282138) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.851041555404663974) ) ) {
                  if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)4.125962495803833896) ) ) {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.142630577087403232) ) ) {
                      result[0] += 0.006923604392640965;
                    } else {
                      result[0] += -0.0031291902721803915;
                    }
                  } else {
                    result[0] += -0.0673931872552306;
                  }
                } else {
                  result[0] += 0.020297039693164674;
                }
              } else {
                if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                  result[0] += -0.05493595602701971;
                } else {
                  result[0] += -0.014740171154263251;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                  if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += 0.09416403971053418;
                  } else {
                    result[0] += -0.06780597755768714;
                  }
                } else {
                  result[0] += 0.1309793503754262;
                }
              } else {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                    result[0] += -0.04587629154954886;
                  } else {
                    if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.357691764831543413) ) ) {
                      result[0] += 0.007087959109853825;
                    } else {
                      result[0] += -0.027654850692076044;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.11162413838586116;
                  } else {
                    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)10.81628036499023615) ) ) {
                        result[0] += 0.033379275766898774;
                      } else {
                        result[0] += -0.04490771733590018;
                      }
                    } else {
                      result[0] += -0.0018826553155331643;
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.2121162414550799) ) ) {
                if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                  if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.511434078216553178) ) ) {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += -0.007168664125316554;
                    } else {
                      result[0] += 0.0322572251513454;
                    }
                  } else {
                    if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.357691764831543413) ) ) {
                      if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
                        result[0] += 0.10826707102001948;
                      } else {
                        result[0] += -5.227055515497861e-06;
                      }
                    } else {
                      result[0] += -0.06573083377114773;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.006301011243631545;
                  } else {
                    result[0] += 0.0029137820416551847;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.431901693344116655) ) ) {
                  result[0] += -0.0336245484187041;
                } else {
                  if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
                    if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
                      result[0] += 0.10163419216673733;
                    } else {
                      result[0] += 0.008679933616695263;
                    }
                  } else {
                    result[0] += 0.0345429095816587;
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.962127923965454546) ) ) {
            result[0] += -0.015701183555927225;
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.531673669815064365) ) ) {
              result[0] += -0.02217629678771874;
            } else {
              result[0] += 0.047526570022637514;
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)8.148167133331300604) ) ) {
      result[0] += -0.11130435483465038;
    } else {
      result[0] += 0.001577746698395622;
    }
  }
  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.056097030639650214) ) ) {
      if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
        result[0] += 0.0006489986215463985;
      } else {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.431880712509156162) ) ) {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.531673669815064365) ) ) {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
              result[0] += -0.06029299448384282;
            } else {
              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += -0.0009943650634389384;
              } else {
                result[0] += 0.017529791461725338;
              }
            }
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.0003618156958216945;
                } else {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.021265694088248202;
                  } else {
                    if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.138333082199097124) ) ) {
                      result[0] += -0.011298186583456273;
                    } else {
                      if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.000000000000000444) ) ) {
                        result[0] += 0.055864474456097546;
                      } else {
                        result[0] += -0.00267274631863321;
                      }
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.040249787734404865;
                } else {
                  result[0] += 0.044892832123347756;
                }
              }
            } else {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.778982400894165927) ) ) {
                  result[0] += -0.06426656038387096;
                } else {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                    if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.068990230560303623) ) ) {
                      result[0] += -0.02471508981102362;
                    } else {
                      result[0] += -0.12038563701508653;
                    }
                  } else {
                    result[0] += 0.0077444777426553995;
                  }
                }
              } else {
                result[0] += -0.06561517157743223;
              }
            }
          }
        } else {
          result[0] += 0.007231568936646864;
        }
      }
    } else {
      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
        result[0] += -0.005618601247561188;
      } else {
        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.959391355514527255) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.91907978057861506) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.088880300521851474) ) ) {
                if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                  result[0] += 0.007296986515005173;
                } else {
                  result[0] += -0.03848851787427959;
                }
              } else {
                result[0] += -0.006295016234295736;
              }
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.888826131820679155) ) ) {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.071567356586456743) ) ) {
                  result[0] += 0.0016920176055591264;
                } else {
                  result[0] += 0.039145503752222935;
                }
              } else {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += -0.005190400606504799;
                } else {
                  result[0] += -0.024855355118888294;
                }
              }
            }
          } else {
            result[0] += -0.03295501062444449;
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.49584054946899592) ) ) {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                result[0] += -0.0038392256410781063;
              } else {
                result[0] += -0.03572710760025556;
              }
            } else {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
                result[0] += 0.006753698112506751;
              } else {
                result[0] += -0.03450711238146167;
              }
            }
          } else {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
              result[0] += 0.0008396876717212369;
            } else {
              if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += 0.009835403728932234;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.05835151672363459) ) ) {
                    result[0] += 0.004264317896535999;
                  } else {
                    result[0] += 0.056984063582183386;
                  }
                }
              } else {
                result[0] += 0.0041932815944612685;
              }
            }
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.20077323913574396) ) ) {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.24173307418823331) ) ) {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
            result[0] += -0.010238915849253266;
          } else {
            result[0] += 0.009147081243394547;
          }
        } else {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += 0.008513088777895327;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.055311203002930576) ) ) {
                result[0] += -0.012414274310977184;
              } else {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                  if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                    result[0] += 0.05110191505027787;
                  } else {
                    result[0] += -0.0006634647846121988;
                  }
                } else {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)1.497866153717041238) ) ) {
                    result[0] += 0.022517499846651495;
                  } else {
                    result[0] += -0.19407367196545927;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.189540147781372958) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.835998296737671787) ) ) {
                result[0] += 0.009817394455272677;
              } else {
                result[0] += 0.04741847856185061;
              }
            } else {
              result[0] += 0.006798569697378818;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.513699531555176669) ) ) {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.23832273483276456) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.617236852645874912) ) ) {
              result[0] += -0.0002814795199906382;
            } else {
              result[0] += -0.013470814498768746;
            }
          } else {
            result[0] += 0.004336592321464049;
          }
        } else {
          if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
              result[0] += -0.058131367624329904;
            } else {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.010952865795918303;
                  } else {
                    result[0] += 0.01211349236861879;
                  }
                } else {
                  result[0] += -0.02394004049267268;
                }
              } else {
                result[0] += 0.0034180527274680858;
              }
            }
          } else {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.962127923965454546) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.029068946838379794) ) ) {
                  result[0] += -0.037682373257083214;
                } else {
                  result[0] += -0.01039277902036055;
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.241523027420044833) ) ) {
                  result[0] += -0.02506859536186571;
                } else {
                  if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                    result[0] += -0.0039574678414473225;
                  } else {
                    result[0] += 0.0566900021462505;
                  }
                }
              }
            } else {
              result[0] += -0.05092971155471443;
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.174569487571716753) ) ) {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
          result[0] += -0.010432183450905543;
        } else {
          result[0] += -0.041714374461944474;
        }
      } else {
        result[0] += 0.0035061569793125506;
      }
    }
  }
  if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)8.071062088012697089) ) ) {
    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
      if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
        result[0] += 0.00019858506111361543;
      } else {
        if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.569433569908142534) ) ) {
          if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.868834793567657693) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.531673669815064365) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.434600353240968573) ) ) {
                  result[0] += -0.03412527560672317;
                } else {
                  result[0] += 0.0073125594282533665;
                }
              } else {
                result[0] += -0.06002961902909472;
              }
            } else {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.43267917633056818) ) ) {
                  result[0] += -0.005338673214804721;
                } else {
                  result[0] += -0.0530635232394511;
                }
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.67046499252319514) ) ) {
                  result[0] += -0.003670520438996073;
                } else {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += -0.02463878671504571;
                  } else {
                    result[0] += 0.04110025203310341;
                  }
                }
              }
            }
          } else {
            result[0] += -0.042830254614659;
          }
        } else {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
            result[0] += 0.0005954194016037305;
          } else {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.53326439857482999) ) ) {
              result[0] += -0.06179655483259951;
            } else {
              result[0] += 0.05738493459092096;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.129040718078614169) ) ) {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
              if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += 0.009025411908619083;
              } else {
                result[0] += 0.11687578045507169;
              }
            } else {
              result[0] += -0.003196235399504561;
            }
          } else {
            result[0] += -0.005061526173492883;
          }
        } else {
          if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.142630577087403232) ) ) {
              if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                result[0] += -0.014320046466243064;
              } else {
                result[0] += 0.07065658343696403;
              }
            } else {
              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += 0.011227418823056457;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)2.138333082199097124) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.216319084167481357) ) ) {
                    result[0] += 0.0006327920871647289;
                  } else {
                    result[0] += 0.056565713725659696;
                  }
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.851041555404663974) ) ) {
                    if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += 0.003824182583868767;
                    } else {
                      result[0] += -0.018393849641167536;
                    }
                  } else {
                    result[0] += -0.03376985876883743;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.255632162094117099) ) ) {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.700598716735840066) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.433569431304932529) ) ) {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                    result[0] += 0.033097176907890415;
                  } else {
                    result[0] += 0.07689245126366484;
                  }
                } else {
                  result[0] += 0.024805359238002703;
                }
              } else {
                result[0] += 0.010081137195772598;
              }
            } else {
              result[0] += -0.0041204349556558705;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.2121162414550799) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.664408206939698154) ) ) {
                  result[0] += 0.12014954916779819;
                } else {
                  if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += -0.007554015527471952;
                  } else {
                    result[0] += 0.0290450481019292;
                  }
                }
              } else {
                if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += 0.01733500901995677;
                } else {
                  result[0] += 0.04772532091473804;
                }
              }
            } else {
              if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
                  result[0] += 0.0034817339048216923;
                } else {
                  result[0] += -0.05653806832098346;
                }
              } else {
                result[0] += -0.03339264411005032;
              }
            }
          } else {
            result[0] += -0.01757610711565273;
          }
        } else {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.36105370521545499) ) ) {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.003321207402191747;
                } else {
                  result[0] += 0.03585317010524634;
                }
              } else {
                if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += -0.041993580672491294;
                } else {
                  result[0] += 0.017351344157199418;
                }
              }
            } else {
              result[0] += -0.00971197688407985;
            }
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.43450713157653853) ) ) {
              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.02751000891394392;
                } else {
                  if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                        result[0] += -0.07640150197795842;
                      } else {
                        result[0] += 0.006551125903473754;
                      }
                    } else {
                      if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += -0.059030994910389524;
                      } else {
                        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.53326439857482999) ) ) {
                              result[0] += -0.07779656928515737;
                            } else {
                              result[0] += 0.014515458472765155;
                            }
                          } else {
                            result[0] += -0.06963210566076711;
                          }
                        } else {
                          result[0] += -0.0757115672911054;
                        }
                      }
                    }
                  } else {
                    result[0] += 0.015379083603315986;
                  }
                }
              } else {
                if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.637949228286744052) ) ) {
                      if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                        result[0] += 0.07059045019314975;
                      } else {
                        result[0] += -0.017519930506230717;
                      }
                    } else {
                      result[0] += -0.0402655196158459;
                    }
                  } else {
                    result[0] += -0.05280399663128417;
                  }
                } else {
                  if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.024461742676747403;
                  } else {
                    result[0] += 0.026773196357819648;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.048819788498816755;
              } else {
                result[0] += 0.094997548258017;
              }
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)8.148167133331300604) ) ) {
      result[0] += -0.11094781077993716;
    } else {
      result[0] += 0.001571996695571733;
    }
  }
  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.56941866874694913) ) ) {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.802901029586792436) ) ) {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.029068946838379794) ) ) {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
              result[0] += 0.0030532884486847026;
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.262283086776734287) ) ) {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += 0.012289141576290867;
                } else {
                  result[0] += -0.034454033354521844;
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.97887301445007413) ) ) {
                  result[0] += -0.005572230495590143;
                } else {
                  result[0] += -0.03981611366353782;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.439304351806642401) ) ) {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.918272972106934482) ) ) {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.105651378631592685) ) ) {
                    result[0] += 0.0655517113250657;
                  } else {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += 0.008532819432334611;
                    } else {
                      result[0] += 0.04260214174568884;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += -0.020235681855594226;
                  } else {
                    result[0] += 0.0018549715571082412;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                  if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += -0.010016933577994868;
                    } else {
                      if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                        result[0] += -0.020751896332646763;
                      } else {
                        result[0] += -0.093492125863309;
                      }
                    }
                  } else {
                    result[0] += -0.14003809437710754;
                  }
                } else {
                  result[0] += -0.0007878909472141578;
                }
              }
            } else {
              if ( UNLIKELY(  (data[38].missing != -1) && (data[38].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                result[0] += 0.00707791874122655;
              } else {
                result[0] += -0.010136562524824682;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
            result[0] += -0.010942311563303298;
          } else {
            result[0] += 0.011189093219008295;
          }
        }
      } else {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
          if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += -0.013291459519087993;
            } else {
              result[0] += 0.0009454002025186758;
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.993164777755738193) ) ) {
              result[0] += 0.0248677078505339;
            } else {
              result[0] += -0.004025105563052851;
            }
          }
        } else {
          result[0] += 0.012558245416640216;
        }
      }
    } else {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
        result[0] += -0.0015266164515018855;
      } else {
        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)8.285748958587648261) ) ) {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.23832273483276456) ) ) {
              result[0] += -0.001809585352633189;
            } else {
              result[0] += -0.020860034960521367;
            }
          } else {
            if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += 0.131354728118427;
            } else {
              result[0] += -0.04588022404991621;
            }
          }
        } else {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.267844915390015537) ) ) {
            if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)7.971558809280396396) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
                result[0] += 0.0018598005060852568;
              } else {
                if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.918272972106934482) ) ) {
                    result[0] += -0.000799417213259396;
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.70586872100830256) ) ) {
                      result[0] += 0.013712739604172776;
                    } else {
                      result[0] += 0.1184624528495764;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += -0.047772153590137575;
                  } else {
                    result[0] += 0.0036279039578746793;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.16554746123882513;
              } else {
                result[0] += 0.011576187944141794;
              }
            }
          } else {
            if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.434600353240968573) ) ) {
                result[0] += 0.007009021801940025;
              } else {
                if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += -0.03433002059872604;
                    } else {
                      result[0] += 0.004707468626909661;
                    }
                  } else {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.249904870986938921) ) ) {
                      result[0] += 0.04175832604571286;
                    } else {
                      if ( UNLIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += 0.021000784567586975;
                      } else {
                        result[0] += -0.0017407372724200215;
                      }
                    }
                  }
                } else {
                  result[0] += -0.046237236123440405;
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.576439857482911933) ) ) {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.513699531555176669) ) ) {
                    result[0] += -0.042090487956606235;
                  } else {
                    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)0.8958797454833985485) ) ) {
                        if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
                          result[0] += 0.11983927575886721;
                        } else {
                          result[0] += -0.0749676106068856;
                        }
                      } else {
                        result[0] += 0.041740285461712934;
                      }
                    } else {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.700598716735840066) ) ) {
                        result[0] += -0.015252715624347643;
                      } else {
                        result[0] += 0.0320893255122928;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                    if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.012675821781158891) ) ) {
                      result[0] += -0.048527328209022176;
                    } else {
                      result[0] += 0.11274532137149543;
                    }
                  } else {
                    result[0] += 0.01788896760748769;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += -0.017865654159155118;
                  } else {
                    if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.010764908219845885;
                    } else {
                      result[0] += 0.05867662860797815;
                    }
                  }
                } else {
                  result[0] += 0.012642194811478453;
                }
              }
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)1.777674019336700661) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.055311203002930576) ) ) {
          result[0] += -0.013337938169738193;
        } else {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.05835151672363459) ) ) {
              result[0] += 0.10212412837345626;
            } else {
              result[0] += 0.008119464489651193;
            }
          } else {
            result[0] += -0.10441730478630969;
          }
        }
      } else {
        result[0] += 0.00339212432119922;
      }
    } else {
      result[0] += 0.00032983100106625146;
    }
  }
  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)7.971558809280396396) ) ) {
    if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
      result[0] += -0.0003774682521272002;
    } else {
      if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
        if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
          result[0] += -0.014393898119247664;
        } else {
          if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.924581527709961826) ) ) {
                  if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.122815132141115058) ) ) {
                    if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.624251961708069292) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.277936458587647373) ) ) {
                        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                          if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
                            result[0] += -0.026037624162510743;
                          } else {
                            result[0] += 0.033428156788556096;
                          }
                        } else {
                          result[0] += 0.026559112842865093;
                        }
                      } else {
                        if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += 0.017055471623110226;
                        } else {
                          result[0] += -0.0151843632890145;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.676220536231995073) ) ) {
                        result[0] += 0.10830585278518284;
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.381086945533752885) ) ) {
                          result[0] += 0.11445879112246796;
                        } else {
                          result[0] += 0.0138380854710082;
                        }
                      }
                    }
                  } else {
                    result[0] += -0.021230037087731038;
                  }
                } else {
                  result[0] += -0.01567504111011979;
                }
              } else {
                if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.605120182037354404) ) ) {
                      result[0] += -0.05303969715175934;
                    } else {
                      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.329314231872559482) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.30853915214538663) ) ) {
                          result[0] += -0.03818692591558973;
                        } else {
                          if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                            if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
                              result[0] += 0.052505043430691206;
                            } else {
                              result[0] += -0.006713538278173063;
                            }
                          } else {
                            result[0] += 0.086037325545252;
                          }
                        }
                      } else {
                        result[0] += 0.06708950030632237;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.868834793567657693) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.94957673549652144) ) ) {
                        result[0] += -0.16340880200337557;
                      } else {
                        if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.223893642425538886) ) ) {
                          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.216319084167481357) ) ) {
                            if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.701225757598877397) ) ) {
                              result[0] += -0.03788635144502474;
                            } else {
                              result[0] += 0.029470844329790746;
                            }
                          } else {
                            result[0] += 0.030927955186321843;
                          }
                        } else {
                          result[0] += -0.0250233118671727;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.780892848968506748) ) ) {
                        result[0] += -0.035009339371635015;
                      } else {
                        result[0] += 0.028805627336463604;
                      }
                    }
                  }
                } else {
                  result[0] += -0.008541159404619945;
                }
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.636499762535095659) ) ) {
                result[0] += 0.030100394212658168;
              } else {
                result[0] += -0.03562403531733504;
              }
            }
          } else {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.766185760498047763) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.397998809814454013) ) ) {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                    if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.665476083755494052) ) ) {
                        result[0] += 0.006704952657460729;
                      } else {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.780892848968506748) ) ) {
                          result[0] += -0.10679965267550923;
                        } else {
                          result[0] += -0.027209086998056454;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += -0.0065629367458039534;
                      } else {
                        result[0] += 0.021998498588989654;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.510617971420288974) ) ) {
                      result[0] += 0.0981010708217886;
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.58491539955139249) ) ) {
                        result[0] += -0.043854516554834105;
                      } else {
                        result[0] += 0.02230594860474787;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1.497866153717041238) ) ) {
                    result[0] += -0.04254029478641504;
                  } else {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
                      result[0] += -0.0033052194406775616;
                    } else {
                      result[0] += -0.10518005629142305;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.851041555404663974) ) ) {
                    if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.019189308410807914;
                    } else {
                      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.088880300521851474) ) ) {
                          result[0] += 0.028611526014713423;
                        } else {
                          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.344550132751465732) ) ) {
                            result[0] += 0.01674944036474728;
                          } else {
                            result[0] += -0.06005176096007672;
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.012675821781158891) ) ) {
                          result[0] += -0.04626822828471537;
                        } else {
                          result[0] += 0.005485312209057253;
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                        result[0] += -0.039145682441278196;
                      } else {
                        result[0] += -0.10618325996632327;
                      }
                    } else {
                      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                        result[0] += -0.03162100735826017;
                      } else {
                        result[0] += 0.04818868107550517;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.848652839660646308) ) ) {
                    if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.02375864199291988;
                    } else {
                      result[0] += -0.01369425035024603;
                    }
                  } else {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.569529533386231357) ) ) {
                        result[0] += 0.04290254823756634;
                      } else {
                        result[0] += -0.049447828230237474;
                      }
                    } else {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.90474271774292081) ) ) {
                        result[0] += -0.015204145313243612;
                      } else {
                        result[0] += 0.07161306156736628;
                      }
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.700598716735840066) ) ) {
                if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                  result[0] += -0.002604068922760839;
                } else {
                  result[0] += 0.03482024087963907;
                }
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.770631790161133257) ) ) {
                  result[0] += -0.0365157008011513;
                } else {
                  if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.515218973159790483) ) ) {
                    result[0] += 0.02267210402493694;
                  } else {
                    result[0] += -0.08039782017162754;
                  }
                }
              }
            }
          }
        }
      } else {
        result[0] += 0.00026112565888389646;
      }
    }
  } else {
    if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)12.00000000000000178) ) ) {
      result[0] += 0.0025658299823067138;
    } else {
      result[0] += -0.16450743366564002;
    }
  }
  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)7.971558809280396396) ) ) {
    if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
          result[0] += -0.03186681059624782;
        } else {
          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)8.285748958587648261) ) ) {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.777633190155030185) ) ) {
                  if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += 0.00464925271677647;
                  } else {
                    result[0] += -0.013036301995608932;
                  }
                } else {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                    if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
                        result[0] += -0.05069120911653427;
                      } else {
                        result[0] += -0.014772168549152045;
                      }
                    } else {
                      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                        result[0] += -0.01928572645268485;
                      } else {
                        result[0] += 0.019690209088254623;
                      }
                    }
                  } else {
                    result[0] += 0.019653206316326152;
                  }
                }
              } else {
                result[0] += -0.030493325538283118;
              }
            } else {
              if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.06551474449051292;
              } else {
                result[0] += 0.1191154256397588;
              }
            }
          } else {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
                if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)2.764714598655701128) ) ) {
                  result[0] += -0.002136611332153852;
                } else {
                  result[0] += 0.17324193267827503;
                }
              } else {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += 0.0021865085491062043;
                  } else {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += 0.0139280355585494;
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.067782521247864214) ) ) {
                        result[0] += -0.049557508119674354;
                      } else {
                        if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                          result[0] += 0.03469102973840884;
                        } else {
                          result[0] += 0.0804281319726089;
                        }
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.302512168884278232) ) ) {
                    result[0] += -0.007245686229136586;
                  } else {
                    result[0] += 0.01600982177259387;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.05318158449515109;
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.778982400894165927) ) ) {
                      if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
                        result[0] += -0.00025470880000668923;
                      } else {
                        result[0] += 0.043143951961012644;
                      }
                    } else {
                      result[0] += -0.004453645645662759;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.637949228286744052) ) ) {
                        if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                          result[0] += 0.057347544973241495;
                        } else {
                          result[0] += -0.009032610787575812;
                        }
                      } else {
                        result[0] += -0.02813578046755081;
                      }
                    } else {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.441542863845826083) ) ) {
                        result[0] += -0.034169132446695616;
                      } else {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                          result[0] += 0.07924660895723248;
                        } else {
                          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.289595603942871982) ) ) {
                            result[0] += 0.009392696767497734;
                          } else {
                            result[0] += 0.08846990563421152;
                          }
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.855006217956543857) ) ) {
                      result[0] += 0.01513987704759434;
                    } else {
                      if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.249904870986938921) ) ) {
                        result[0] += 0.02666757404135521;
                      } else {
                        result[0] += 0.09382279360151795;
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.662244915962219682) ) ) {
                  if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5.400420188903809482) ) ) {
                    result[0] += 0.006920552452694628;
                  } else {
                    if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)3.000000000000000444) ) ) {
                      result[0] += -0.03315203067702224;
                    } else {
                      result[0] += 0.10519123310757575;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)3.000000000000000444) ) ) {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                      if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                        if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                          result[0] += -0.01592129447898829;
                        } else {
                          result[0] += -0.05029410805726941;
                        }
                      } else {
                        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.740319490432739702) ) ) {
                          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.637949228286744052) ) ) {
                            if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.012675821781158891) ) ) {
                                result[0] += -0.07432283579235113;
                              } else {
                                result[0] += 0.0023174215211380095;
                              }
                            } else {
                              result[0] += 0.0016748190020001573;
                            }
                          } else {
                            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                              result[0] += -0.10197844802507057;
                            } else {
                              result[0] += -0.015751396864593033;
                            }
                          }
                        } else {
                          if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
                            if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)7.508512496948243076) ) ) {
                              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
                                result[0] += -0.007123333518703538;
                              } else {
                                result[0] += -0.05269526293621505;
                              }
                            } else {
                              result[0] += 0.1486081879594289;
                            }
                          } else {
                            if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)4.516324043273926669) ) ) {
                              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.285887241363526279) ) ) {
                                result[0] += -0.07939195861535504;
                              } else {
                                result[0] += 0.01634825629356058;
                              }
                            } else {
                              result[0] += 0.021370499203420622;
                            }
                          }
                        }
                      }
                    } else {
                      result[0] += 0.016997516555881328;
                    }
                  } else {
                    result[0] += -0.004091277627868542;
                  }
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
              result[0] += 0.06625085124393626;
            } else {
              result[0] += -0.025180539829717352;
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.737386107444763628) ) ) {
              result[0] += 0.03590564248619867;
            } else {
              result[0] += -0.04346439903144747;
            }
          }
        } else {
          result[0] += 0.00020024008045344167;
        }
      }
    } else {
      if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
        if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.851041555404663974) ) ) {
            result[0] += -0.02180480821234136;
          } else {
            result[0] += 0.02892605878150119;
          }
        } else {
          result[0] += 0.003984926741777978;
        }
      } else {
        result[0] += 0.00023740026777098933;
      }
    }
  } else {
    if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)12.00000000000000178) ) ) {
      result[0] += 0.0018964567808328555;
    } else {
      result[0] += -0.15873402202273224;
    }
  }
  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)7.971558809280396396) ) ) {
    if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)8.071062088012697089) ) ) {
      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
        if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.029068946838379794) ) ) {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.868834793567657693) ) ) {
                  result[0] += 0.0028142707033493833;
                } else {
                  result[0] += -0.015271082160974173;
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.766185760498047763) ) ) {
                  result[0] += 0.0009317952988045844;
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.029068946838379794) ) ) {
                    result[0] += -0.0047533118979765435;
                  } else {
                    result[0] += -0.030880221714920737;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.020127415657043901) ) ) {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.325443029403687412) ) ) {
                  result[0] += 0.001464602220725975;
                } else {
                  if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                    result[0] += -0.01735839203701579;
                  } else {
                    result[0] += 0.008128823915187102;
                  }
                }
              } else {
                result[0] += -0.006112214080974669;
              }
            }
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
              result[0] += -0.0012509991761564822;
            } else {
              result[0] += 0.0022952870758978744;
            }
          }
        } else {
          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
            if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
              if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)6.103675842285157138) ) ) {
                result[0] += -0.02029461920935575;
              } else {
                result[0] += 0.044512437425479695;
              }
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.597323656082154208) ) ) {
                      result[0] += 0.04293467720113739;
                    } else {
                      result[0] += -0.13610358700965383;
                    }
                  } else {
                    result[0] += -0.0008960305025215115;
                  }
                } else {
                  if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.0006104850298589355;
                  } else {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)8.285748958587648261) ) ) {
                      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                        result[0] += -0.04331016119514209;
                      } else {
                        result[0] += 0.010586832181213581;
                      }
                    } else {
                      result[0] += 0.20477686071877121;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.940167903900147373) ) ) {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                    result[0] += -0.00578328965005057;
                  } else {
                    if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.379217386245728427) ) ) {
                        result[0] += 0.07530273275346411;
                      } else {
                        if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                          result[0] += 0.057135718921856496;
                        } else {
                          result[0] += -0.0444953353334408;
                        }
                      }
                    } else {
                      result[0] += 0.006808366315414489;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.242453336715698464) ) ) {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.733271598815919745) ) ) {
                        result[0] += 0.012624142901284384;
                      } else {
                        result[0] += -0.034050152807592;
                      }
                    } else {
                      if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                        result[0] += -0.010720809015025472;
                      } else {
                        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += 0.053424114040690064;
                        } else {
                          result[0] += -0.009996774882582967;
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.637949228286744052) ) ) {
                      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.40000796318054288) ) ) {
                            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.23832273483276456) ) ) {
                              result[0] += 0.0027929568554558824;
                            } else {
                              result[0] += -0.0422787775784051;
                            }
                          } else {
                            result[0] += 0.01249645217661381;
                          }
                        } else {
                          result[0] += -0.0458704109808126;
                        }
                      } else {
                        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.51693725585937678) ) ) {
                          if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                            result[0] += 0.015749316133229378;
                          } else {
                            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.025192260742188388) ) ) {
                              result[0] += 0.09574394587200574;
                            } else {
                              result[0] += 0.03495379248151407;
                            }
                          }
                        } else {
                          result[0] += -0.04308948505776988;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                        if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)7.112327098846436435) ) ) {
                          result[0] += 0.007767735038033155;
                        } else {
                          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                            result[0] += -0.03146021327786732;
                          } else {
                            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                                result[0] += -0.007920071627241796;
                              } else {
                                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
                                  result[0] += 0.07621393512821253;
                                } else {
                                  result[0] += -0.008689384685431874;
                                }
                              }
                            } else {
                              result[0] += -0.027287606761875173;
                            }
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                            result[0] += -0.07697784013330794;
                          } else {
                            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.90474271774292081) ) ) {
                              result[0] += -0.05509528459855642;
                            } else {
                              result[0] += 0.12495295946689232;
                            }
                          }
                        } else {
                          if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.701225757598877397) ) ) {
                            result[0] += -0.002473742586904309;
                          } else {
                            result[0] += -0.056749646043798264;
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          } else {
            result[0] += 0.00025981283347480073;
          }
        }
      } else {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.655405282974244052) ) ) {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
            result[0] += -0.08888845394328365;
          } else {
            if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.384246587753296343) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.94957673549652144) ) ) {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.249904870986938921) ) ) {
                  result[0] += 0.04078332043149635;
                } else {
                  result[0] += -0.021287039086004815;
                }
              } else {
                if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.750972747802735263) ) ) {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.35306882858276456) ) ) {
                      result[0] += 0.014201182958757625;
                    } else {
                      result[0] += -0.057497010900727265;
                    }
                  } else {
                    result[0] += -0.02069573583396793;
                  }
                } else {
                  result[0] += -0.023329679115154028;
                }
              }
            } else {
              result[0] += -0.05172414845207956;
            }
          }
        } else {
          result[0] += 0.096157697761176;
        }
      }
    } else {
      if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)8.148167133331300604) ) ) {
        result[0] += -0.11078416518613499;
      } else {
        result[0] += 0.0018591275538110152;
      }
    }
  } else {
    if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)12.00000000000000178) ) ) {
      result[0] += -0.0011121860600617449;
    } else {
      result[0] += -0.1483342314310221;
    }
  }
  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)7.971558809280396396) ) ) {
    if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)8.071062088012697089) ) ) {
      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
            if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
              if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.102609157562256748) ) ) {
                if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.00551257432836173;
                } else {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
                      if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += -0.01703744838738094;
                      } else {
                        result[0] += -0.06753959783314525;
                      }
                    } else {
                      result[0] += 0.009639502486546004;
                    }
                  } else {
                    if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.164715528488160068) ) ) {
                        result[0] += -0.02849018552444888;
                      } else {
                        result[0] += 0.015097248760692808;
                      }
                    } else {
                      if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.138333082199097124) ) ) {
                        result[0] += -0.10624184866774296;
                      } else {
                        result[0] += -0.04862915513827582;
                      }
                    }
                  }
                }
              } else {
                result[0] += 0.0014916917119883423;
              }
            } else {
              result[0] += 0.0077061572441220905;
            }
          } else {
            if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.700598716735840066) ) ) {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.0006367430206821059;
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.53326439857482999) ) ) {
                  result[0] += 0.021840262747749663;
                } else {
                  result[0] += -0.0016112121063139172;
                }
              }
            } else {
              result[0] += -0.018787903291164518;
            }
          }
        } else {
          result[0] += 0.0008930309692520072;
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.190353393554689276) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.164715528488160068) ) ) {
            if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)4.125962495803833896) ) ) {
              if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.0008375759728199513;
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.357691764831543413) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.909254074096680576) ) ) {
                    result[0] += -0.017296456531762292;
                  } else {
                    result[0] += 0.09855741416242166;
                  }
                } else {
                  if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)47227863040.00000763) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.288254261016846591) ) ) {
                      if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                        result[0] += -0.0037946039950810815;
                      } else {
                        result[0] += 0.039082362032869744;
                      }
                    } else {
                      result[0] += -0.018761339194242;
                    }
                  } else {
                    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                      if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.255632162094117099) ) ) {
                          if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)0.8958797454833985485) ) ) {
                            result[0] += -0.008020089972328212;
                          } else {
                            result[0] += 0.0271431607738723;
                          }
                        } else {
                          result[0] += -0.0009554667289541715;
                        }
                      } else {
                        if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.219419956207276279) ) ) {
                            result[0] += -0.0017999258040980793;
                          } else {
                            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.674522399902344638) ) ) {
                              result[0] += -0.06148520288804987;
                            } else {
                              result[0] += -0.01159794818445506;
                            }
                          }
                        } else {
                          result[0] += 0.01612554384572703;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                        result[0] += 0.007883569347382794;
                      } else {
                        if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
                          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.467917680740357333) ) ) {
                            result[0] += 0.03777616996806176;
                          } else {
                            result[0] += 0.00400975587392737;
                          }
                        } else {
                          if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.012675821781158891) ) ) {
                            result[0] += 0.0003025119306866425;
                          } else {
                            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                              result[0] += 0.02619223050702832;
                            } else {
                              if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.474771499633789951) ) ) {
                                  result[0] += 0.05996538437961259;
                                } else {
                                  result[0] += -0.005452072061122627;
                                }
                              } else {
                                result[0] += 0.1828036168524682;
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
              result[0] += -0.04856057923259177;
            }
          } else {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.511434078216553178) ) ) {
                if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += 0.0009329083764655606;
                } else {
                  result[0] += 0.04764449437092119;
                }
              } else {
                result[0] += -0.01478057348851363;
              }
            } else {
              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)2.740319490432739702) ) ) {
                  result[0] += -0.06460827484150675;
                } else {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += 0.004444138694671365;
                  } else {
                    result[0] += 0.026881083734656933;
                  }
                }
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.617236852645874912) ) ) {
                  result[0] += -0.0163242052665805;
                } else {
                  result[0] += 0.031143485093728596;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.637949228286744052) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.01293420791626154) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
                result[0] += 0.012278602127129829;
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.835998296737671787) ) ) {
                  result[0] += -0.013105689908139099;
                } else {
                  result[0] += 0.0007931740210214013;
                }
              }
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.761470437049866167) ) ) {
                result[0] += -0.023092562549898502;
              } else {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)1.497866153717041238) ) ) {
                  result[0] += 0.00972061396456842;
                } else {
                  result[0] += -0.061638260628391006;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.90474271774292081) ) ) {
              if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.0857105340154943;
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.801954269409180576) ) ) {
                    result[0] += -0.058473495673732626;
                  } else {
                    result[0] += 0.05035500493493127;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += -0.011072928892730306;
                } else {
                  result[0] += -0.02372593252804804;
                }
              }
            } else {
              result[0] += 0.00023008041574264665;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)8.148167133331300604) ) ) {
        result[0] += -0.11078416518613499;
      } else {
        result[0] += 0.0018591275538110152;
      }
    }
  } else {
    if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
        result[0] += -0.13418430469256612;
      } else {
        result[0] += 0.009256515592763208;
      }
    } else {
      if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
        result[0] += 0.009565771628244596;
      } else {
        result[0] += -0.19555126947313722;
      }
    }
  }
  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)15.20015096664428889) ) ) {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.174569487571716753) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.567862033843995029) ) ) {
          if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += 0.0005077359240276358;
          } else {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.924581527709961826) ) ) {
                result[0] += -0.0829832493180922;
              } else {
                result[0] += 0.0005822293230872728;
              }
            } else {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.384830474853516513) ) ) {
                result[0] += -0.003795587342291314;
              } else {
                if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
                  result[0] += -0.005123582603619303;
                } else {
                  result[0] += -0.04846921831030541;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.0036626789777294954;
            } else {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += -0.001183275329305514;
              } else {
                if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.802901029586792436) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.270308971405030185) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.262283086776734287) ) ) {
                      result[0] += -0.05313382304235836;
                    } else {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.14301252365112482) ) ) {
                        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.05835151672363459) ) ) {
                          result[0] += -0.004533739690505562;
                        } else {
                          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.53326439857482999) ) ) {
                            result[0] += -0.008921516054302988;
                          } else {
                            result[0] += -0.05003083672371264;
                          }
                        }
                      } else {
                        result[0] += 0.006486822126394717;
                      }
                    }
                  } else {
                    result[0] += 0.010034348395998198;
                  }
                } else {
                  if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.262283086776734287) ) ) {
                        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.2531323432922381) ) ) {
                          result[0] += 0.0038909669774894714;
                        } else {
                          result[0] += -0.0437786937758718;
                        }
                      } else {
                        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
                          if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                            result[0] += 0.07422149566572235;
                          } else {
                            result[0] += 0.020348635280212368;
                          }
                        } else {
                          result[0] += -0.016496148698519238;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
                        if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.497866153717041238) ) ) {
                          result[0] += -0.012602420726562542;
                        } else {
                          result[0] += 0.10002643790217228;
                        }
                      } else {
                        result[0] += 0.010607143046795173;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.623839378356934482) ) ) {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.865389823913576) ) ) {
                        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                          result[0] += -0.0014454506286468202;
                        } else {
                          result[0] += -0.027399845271586145;
                        }
                      } else {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.262283086776734287) ) ) {
                          result[0] += -0.003954987510270976;
                        } else {
                          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.9055976867675799) ) ) {
                            result[0] += 0.01723275020595785;
                          } else {
                            result[0] += 0.047596994594844996;
                          }
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += 0.03580211770663292;
                      } else {
                        result[0] += 0.013941582849304916;
                      }
                    }
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.914472818374634233) ) ) {
                result[0] += 0.11809736646199061;
              } else {
                result[0] += -0.016225122659325596;
              }
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.76779222488403498) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.344550132751465732) ) ) {
                    result[0] += -0.01229911546198341;
                  } else {
                    result[0] += 0.020518512616416695;
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.802696108818054643) ) ) {
                    result[0] += 0.05164915219736178;
                  } else {
                    result[0] += -0.04672210269877225;
                  }
                }
              } else {
                result[0] += 0.03676416979394062;
              }
            }
          }
        }
      } else {
        result[0] += -0.006969409110838132;
      }
    } else {
      result[0] += 0.01209736241935036;
    }
  } else {
    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.16594791412353693) ) ) {
      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.921924352645874468) ) ) {
          result[0] += 0.011305192454889541;
        } else {
          if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.665476083755494052) ) ) {
              result[0] += -0.022624825759004465;
            } else {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += 0.01731008713719297;
              } else {
                result[0] += -0.018852111960503487;
              }
            }
          } else {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.637949228286744052) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.342454433441162998) ) ) {
                result[0] += 0.01879847158454291;
              } else {
                result[0] += -0.021480006545494146;
              }
            } else {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += 0.07406538663249303;
                } else {
                  result[0] += 0.01652797177123006;
                }
              } else {
                result[0] += 0.004795910748695897;
              }
            }
          }
        }
      } else {
        result[0] += 0.0001679203375540483;
      }
    } else {
      if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.497866153717041238) ) ) {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.737386107444763628) ) ) {
            result[0] += -0.019603189866090852;
          } else {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.384830474853516513) ) ) {
                  if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.006322713519826103;
                  } else {
                    result[0] += -0.025655857239202697;
                  }
                } else {
                  result[0] += 0.016449582963468875;
                }
              } else {
                result[0] += 0.006163044809443992;
              }
            } else {
              if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.219419956207276279) ) ) {
                  result[0] += -0.015253078752656805;
                } else {
                  result[0] += 0.00420613918348516;
                }
              } else {
                if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                  result[0] += 0.040428648181579534;
                } else {
                  if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += -0.000703241666251576;
                  } else {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.071567356586456743) ) ) {
                      result[0] += 0.03810456943081595;
                    } else {
                      result[0] += -0.029288968704987452;
                    }
                  }
                }
              }
            }
          }
        } else {
          result[0] += -0.024554496171226244;
        }
      } else {
        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
          result[0] += 0.0035510857862775832;
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.53326439857482999) ) ) {
            result[0] += 0.0016005986724107518;
          } else {
            result[0] += 0.11001044111129188;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)7.971558809280396396) ) ) {
    if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.000000000000000444) ) ) {
      if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)0.8958797454833985485) ) ) {
        if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
          if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.265274047851563388) ) ) {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.373361587524414951) ) ) {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.233438730239869052) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.255632162094117099) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.046861171722413886) ) ) {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.861792564392090288) ) ) {
                        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.142630577087403232) ) ) {
                          result[0] += 0.09069739417944794;
                        } else {
                          result[0] += 0.008930354216695992;
                        }
                      } else {
                        if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += 0.039667962565363264;
                        } else {
                          result[0] += 0.13530519202921762;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.740319490432739702) ) ) {
                        result[0] += -0.09879107117773918;
                      } else {
                        result[0] += 0.022615025114224075;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.467917680740357333) ) ) {
                        if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.980170249938965732) ) ) {
                          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.993164777755738193) ) ) {
                              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.5240359306335467) ) ) {
                                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.941534638404846635) ) ) {
                                  result[0] += 0.003358237111128937;
                                } else {
                                  result[0] += 0.04146256502525733;
                                }
                              } else {
                                result[0] += -0.025347885714773616;
                              }
                            } else {
                              result[0] += -0.045197955992956056;
                            }
                          } else {
                            if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.01634240150451749) ) ) {
                              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.921060562133789951) ) ) {
                                if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                                  result[0] += -0.022305781288228254;
                                } else {
                                  if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.970085620880127397) ) ) {
                                    result[0] += 0.09051593791769863;
                                  } else {
                                    result[0] += -0.006219241125499389;
                                  }
                                }
                              } else {
                                result[0] += 0.060426113150478106;
                              }
                            } else {
                              result[0] += 0.06816802890002609;
                            }
                          }
                        } else {
                          result[0] += -0.07471586184603819;
                        }
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.744487762451173651) ) ) {
                          result[0] += -0.0491264116136118;
                        } else {
                          if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.602003335952759233) ) ) {
                            result[0] += 0.13200383813013428;
                          } else {
                            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.662244915962219682) ) ) {
                              result[0] += -0.06246443299141531;
                            } else {
                              result[0] += -0.0008723631081058368;
                            }
                          }
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.088880300521851474) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.764287948608400214) ) ) {
                          result[0] += -0.18457389684453618;
                        } else {
                          result[0] += -0.04608978404926007;
                        }
                      } else {
                        result[0] += 0.00033289541587791624;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.12713098526001154) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.05835151672363459) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.780892848968506748) ) ) {
                        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.046861171722413886) ) ) {
                          result[0] += 0.02018182717700845;
                        } else {
                          result[0] += 0.08124132461118694;
                        }
                      } else {
                        result[0] += 0.004656342987089405;
                      }
                    } else {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.938058137893677646) ) ) {
                        result[0] += -0.1309186544521548;
                      } else {
                        result[0] += -0.01499925602345616;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += 0.017475586184340638;
                    } else {
                      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.23832273483276456) ) ) {
                        result[0] += 0.11442049924553224;
                      } else {
                        result[0] += 0.017236603892053736;
                      }
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.673553824424744096) ) ) {
                  result[0] += 0.008760274258810968;
                } else {
                  if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.11058931520599424;
                  } else {
                    result[0] += -0.015154469342123259;
                  }
                }
              }
            } else {
              result[0] += 0.02846172778001202;
            }
          } else {
            if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.519456863403321201) ) ) {
              result[0] += 0.07881614479632844;
            } else {
              result[0] += 0.017882869907912574;
            }
          }
        } else {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.72561454772949396) ) ) {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.449861526489258257) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.921924352645874468) ) ) {
                result[0] += -0.08243225230057492;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.94957673549652144) ) ) {
                  result[0] += -0.07273582706850325;
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.662244915962219682) ) ) {
                    result[0] += -0.023268103147849065;
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.764287948608400214) ) ) {
                      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.029068946838379794) ) ) {
                          result[0] += 0.07172987216065002;
                        } else {
                          result[0] += 0.008862182923561297;
                        }
                      } else {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.743881702423096591) ) ) {
                          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.909102678298951083) ) ) {
                            result[0] += 0.041870167011381104;
                          } else {
                            result[0] += -0.02740481273596289;
                          }
                        } else {
                          if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
                            result[0] += -0.001015611828089954;
                          } else {
                            result[0] += -0.06451402008135515;
                          }
                        }
                      }
                    } else {
                      result[0] += 0.052807418561309244;
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1.497866153717041238) ) ) {
                result[0] += 0.03588578340083493;
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.158952236175537998) ) ) {
                  if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.029068946838379794) ) ) {
                    result[0] += -0.02711184018794341;
                  } else {
                    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.2001411386386064;
                    } else {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.921924352645874468) ) ) {
                        result[0] += -0.08894348641038022;
                      } else {
                        result[0] += 0.013229817801875841;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.58961367607116788) ) ) {
                    result[0] += -0.002870323619784948;
                  } else {
                    result[0] += -0.06644029881988837;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)7.927865505218506748) ) ) {
              result[0] += 0.06307995795005207;
            } else {
              result[0] += -0.005537660860222841;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.802696108818054643) ) ) {
          if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.855921268463135654) ) ) {
            result[0] += 0.019805353037000723;
          } else {
            result[0] += -0.13433789868847454;
          }
        } else {
          if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.198464870452881303) ) ) {
            result[0] += 0.026648045305825815;
          } else {
            result[0] += 0.10434181708557744;
          }
        }
      }
    } else {
      result[0] += -0.00014333767264542154;
    }
  } else {
    if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)12.00000000000000178) ) ) {
      result[0] += -0.0016882725101830436;
    } else {
      result[0] += 0.16269862732212093;
    }
  }
  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)7.971558809280396396) ) ) {
    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
      result[0] += 0.0005003630467493487;
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.513699531555176669) ) ) {
        result[0] += 0.0006249211115211256;
      } else {
        if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
            result[0] += -0.02642826841360439;
          } else {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.321723937988282138) ) ) {
                  if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += -0.017596318729500078;
                  } else {
                    if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += 0.01225224052826446;
                    } else {
                      result[0] += -0.01216003524171463;
                    }
                  }
                } else {
                  result[0] += 0.028996437599719635;
                }
              } else {
                result[0] += -0.03845978396250965;
              }
            } else {
              if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.959391355514527255) ) ) {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.493027687072754794) ) ) {
                        if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.357691764831543413) ) ) {
                          result[0] += -0.002858733057214863;
                        } else {
                          result[0] += 0.0727767967682696;
                        }
                      } else {
                        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.344550132751465732) ) ) {
                          result[0] += -0.000244941411267635;
                        } else {
                          if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                            result[0] += -0.055339143385339684;
                          } else {
                            result[0] += -0.020625171718743607;
                          }
                        }
                      }
                    } else {
                      result[0] += 0.007267475541775774;
                    }
                  } else {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.182021141052246982) ) ) {
                      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                        result[0] += 0.020640594984521567;
                      } else {
                        result[0] += -0.013649607495185616;
                      }
                    } else {
                      result[0] += -0.01309855874749372;
                    }
                  }
                } else {
                  result[0] += -0.03128963393165532;
                }
              } else {
                if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.602003335952759233) ) ) {
                      result[0] += -0.017282748596598924;
                    } else {
                      if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.676220536231995073) ) ) {
                        result[0] += 0.00037480592623024634;
                      } else {
                        result[0] += 0.015774864680820396;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.379217386245728427) ) ) {
                      if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += -0.015258816650702273;
                      } else {
                        result[0] += -0.05845312106231154;
                      }
                    } else {
                      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                          result[0] += 0.003969379692324615;
                        } else {
                          if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            result[0] += -0.04492551837809705;
                          } else {
                            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                              result[0] += 0.017445395251680467;
                            } else {
                              result[0] += -0.05412769410777708;
                            }
                          }
                        }
                      } else {
                        result[0] += 0.053262251376459326;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.05835151672363459) ) ) {
                        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                          result[0] += -0.003078491949314519;
                        } else {
                          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                              result[0] += -0.07157956243595658;
                            } else {
                              result[0] += 0.026008181194270244;
                            }
                          } else {
                            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.524927973747253862) ) ) {
                              result[0] += 0.025998060135286508;
                            } else {
                              result[0] += -0.03171121711498717;
                            }
                          }
                        }
                      } else {
                        result[0] += 0.017555071304628335;
                      }
                    } else {
                      result[0] += 0.012795132811995178;
                    }
                  } else {
                    result[0] += 0.02631524697213863;
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
            if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
              if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.0348305315501303;
                } else {
                  result[0] += 0.019933764427750723;
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.14301252365112482) ) ) {
                    if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.938867926597595659) ) ) {
                          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                            result[0] += -0.0103537086469564;
                          } else {
                            result[0] += 0.13350208019647808;
                          }
                        } else {
                          if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.32411074638366788) ) ) {
                            result[0] += 0.010992487662373358;
                          } else {
                            result[0] += -0.03700629860918186;
                          }
                        }
                      } else {
                        result[0] += 0.05477967360169275;
                      }
                    } else {
                      result[0] += -0.003887279304889647;
                    }
                  } else {
                    result[0] += -0.012580781515094914;
                  }
                } else {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.909855604171753818) ) ) {
                    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += -0.005702028525956827;
                    } else {
                      if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.877672910690308505) ) ) {
                          result[0] += -0.018268605463059837;
                        } else {
                          result[0] += 0.04675768487134291;
                        }
                      } else {
                        result[0] += -0.04339326487974152;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                        result[0] += -0.0034558599283825483;
                      } else {
                        if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
                          result[0] += 0.02744739690870117;
                        } else {
                          result[0] += -0.02145628695091486;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                        if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                            result[0] += -0.06783012646399561;
                          } else {
                            result[0] += 0.0062717828701878905;
                          }
                        } else {
                          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.56941866874694913) ) ) {
                            result[0] += 0.01650575332538185;
                          } else {
                            result[0] += 0.056121984811081355;
                          }
                        }
                      } else {
                        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                          result[0] += 0.0007972822171029326;
                        } else {
                          result[0] += -0.05511387941078991;
                        }
                      }
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.23832273483276456) ) ) {
                result[0] += 0.005388068245297681;
              } else {
                result[0] += 0.03815560330911368;
              }
            }
          } else {
            result[0] += -0.03901535818168689;
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)12.00000000000000178) ) ) {
      result[0] += 0.00017384979167970672;
    } else {
      result[0] += 0.15722716146660873;
    }
  }
  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)1.242453336715698464) ) ) {
      result[0] += 0.0004364129508628213;
    } else {
      if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)12.00000000000000178) ) ) {
        result[0] += 0.013539162385127704;
      } else {
        result[0] += 0.11667184297874655;
      }
    }
  } else {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.056097030639650214) ) ) {
      if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.825422286987305576) ) ) {
          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += 0.011606009851097054;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.129040718078614169) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
                  result[0] += -0.0291359317301733;
                } else {
                  result[0] += 0.009935547219813271;
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.060294389724732333) ) ) {
                  result[0] += 0.02291568272162564;
                } else {
                  result[0] += -0.00786432856216034;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.302512168884278232) ) ) {
                result[0] += -0.0005509544013036957;
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.745876312255860263) ) ) {
                  result[0] += -0.04715140250405176;
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.433652400970459873) ) ) {
                    result[0] += -0.02347436476925494;
                  } else {
                    result[0] += 0.041938755151708286;
                  }
                }
              }
            } else {
              result[0] += 0.004177071386208555;
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.556798219680787021) ) ) {
            result[0] += 0.0029329930966569755;
          } else {
            if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)6.000000000000000888) ) ) {
              if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.007380646704833136;
                } else {
                  result[0] += 0.016891494559566365;
                }
              } else {
                result[0] += -0.026973274106093372;
              }
            } else {
              if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.020170337484342122;
                } else {
                  result[0] += -0.04939530228050151;
                }
              } else {
                result[0] += 0.012920975686428133;
              }
            }
          }
        }
      } else {
        result[0] += -0.0008075909294770014;
      }
    } else {
      if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
        if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)1.497866153717041238) ) ) {
            result[0] += -0.054799798161707686;
          } else {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.531673669815064365) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.23832273483276456) ) ) {
                result[0] += 0.0025745230620731197;
              } else {
                result[0] += 0.14014810137932068;
              }
            } else {
              if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.0025224909698177934;
              } else {
                if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.718933820724488193) ) ) {
                    if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.497866153717041238) ) ) {
                      result[0] += -0.010260661042492597;
                    } else {
                      if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)12.00000000000000178) ) ) {
                        result[0] += -0.07451953778786782;
                      } else {
                        result[0] += 0.026602404266306626;
                      }
                    }
                  } else {
                    result[0] += -0.0480246918170631;
                  }
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.302512168884278232) ) ) {
                    result[0] += -0.029772729128473636;
                  } else {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += 0.04862823797083089;
                    } else {
                      result[0] += -0.015071000628708343;
                    }
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.262283086776734287) ) ) {
              result[0] += 0.00704186681620067;
            } else {
              if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += 0.0374303702633979;
              } else {
                result[0] += -0.04437720856773552;
              }
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.636499762535095659) ) ) {
              result[0] += -0.05318721723871082;
            } else {
              result[0] += 0.004603245704053798;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.174569487571716753) ) ) {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
              if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.003066174831275318;
                } else {
                  result[0] += -0.03167292311456163;
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.433569431304932529) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.947818994522095615) ) ) {
                    result[0] += -0.0003784675206961322;
                  } else {
                    result[0] += -0.054590164825189996;
                  }
                } else {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                    if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                      result[0] += 0.005437451289660156;
                    } else {
                      result[0] += -0.04729598005350197;
                    }
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.507949829101563388) ) ) {
                      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                        result[0] += 0.008298452162500946;
                      } else {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.397998809814454013) ) ) {
                          result[0] += -0.0109387707916693;
                        } else {
                          result[0] += 0.05225602416508314;
                        }
                      }
                    } else {
                      result[0] += 0.07524973224371442;
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.938867926597595659) ) ) {
                result[0] += 0.03128087225909065;
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.424685239791871005) ) ) {
                  if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += -0.003640211736680258;
                    } else {
                      result[0] += -0.04684110244610274;
                    }
                  } else {
                    result[0] += -0.009738521158485192;
                  }
                } else {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += -0.005414751160090813;
                  } else {
                    if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += 0.000289403051627047;
                    } else {
                      if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += -0.07116709047286929;
                      } else {
                        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.255632162094117099) ) ) {
                          result[0] += -0.0321622134460519;
                        } else {
                          if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                            result[0] += -0.01793632265248508;
                          } else {
                            if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                              result[0] += 0.0005278709548362678;
                            } else {
                              result[0] += 0.097634334590869;
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
            result[0] += -0.04750252948328846;
          }
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.801954269409180576) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.770631790161133257) ) ) {
              result[0] += 0.15719808940459656;
            } else {
              result[0] += 0.0033490848150736032;
            }
          } else {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += -0.004879056090735693;
            } else {
              result[0] += 0.05629555715766155;
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)7.971558809280396396) ) ) {
    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.055496215820313388) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.567862033843995029) ) ) {
          if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.085941076278687412) ) ) {
            if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += 0.018732931362702807;
              } else {
                result[0] += -0.004165970887020979;
              }
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.397998809814454013) ) ) {
                result[0] += -0.0597259875254123;
              } else {
                result[0] += -0.004568433978493987;
              }
            }
          } else {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.15130625149881838;
              } else {
                result[0] += -0.03479381240485766;
              }
            } else {
              result[0] += -0.01869856343737068;
            }
          }
        } else {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
            result[0] += 0.0006676444296699062;
          } else {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.637949228286744052) ) ) {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.700598716735840066) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.029068946838379794) ) ) {
                  if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2415.000000000000455) ) ) {
                    result[0] += -0.010378971316578285;
                  } else {
                    result[0] += -0.06666862684439871;
                  }
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.49241352081299006) ) ) {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.636499762535095659) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.744487762451173651) ) ) {
                        result[0] += -0.05750713293281391;
                      } else {
                        result[0] += 0.06908030971283602;
                      }
                    } else {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.851041555404663974) ) ) {
                        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.2121162414550799) ) ) {
                          if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                            result[0] += 0.018107598240148383;
                          } else {
                            result[0] += -0.026129094846683666;
                          }
                        } else {
                          result[0] += -0.08755429977341017;
                        }
                      } else {
                        result[0] += -0.07290755475783153;
                      }
                    }
                  } else {
                    result[0] += 0.05472624270788822;
                  }
                }
              } else {
                result[0] += -0.07621546311525963;
              }
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.795762062072754794) ) ) {
                result[0] += -0.09369065439288374;
              } else {
                result[0] += -0.03368381941274145;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
          if ( UNLIKELY(  (data[37].missing != -1) && (data[37].fvalue <= (double)-1.00000001800250948e-35) ) ) {
            result[0] += 0.003607461433288027;
          } else {
            if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.321723937988282138) ) ) {
                if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.431880712509156162) ) ) {
                    result[0] += -0.0014144879954301143;
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.801954269409180576) ) ) {
                      result[0] += -0.006684262622758958;
                    } else {
                      result[0] += -0.03487390217976207;
                    }
                  }
                } else {
                  result[0] += 0.011400614839772645;
                }
              } else {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.182021141052246982) ) ) {
                    result[0] += -0.007107911047792077;
                  } else {
                    result[0] += -0.04633186048944132;
                  }
                } else {
                  result[0] += -0.09275822182597719;
                }
              }
            } else {
              if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.709793567657472479) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.623839378356934482) ) ) {
                    result[0] += 0.01255790237626457;
                  } else {
                    result[0] += -0.08599992955710418;
                  }
                } else {
                  if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += -0.07502200735332588;
                  } else {
                    result[0] += -0.02629787525690417;
                  }
                }
              } else {
                result[0] += -0.007980802866849143;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += 0.06114363631978509;
              } else {
                result[0] += 0.0004971014067734627;
              }
            } else {
              if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.883387088775636542) ) ) {
                  result[0] += -0.04490998104766212;
                } else {
                  result[0] += -0.0052587316416598615;
                }
              } else {
                result[0] += 0.016733952487114893;
              }
            }
          } else {
            if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.449861526489258257) ) ) {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.288254261016846591) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.628996372222901279) ) ) {
                    result[0] += -0.012622381988718082;
                  } else {
                    result[0] += 0.06867001955380359;
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.780892848968506748) ) ) {
                    result[0] += 0.0019492325403132967;
                  } else {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
                      result[0] += 0.016454134442700854;
                    } else {
                      result[0] += -0.0473180426993746;
                    }
                  }
                }
              } else {
                result[0] += 0.0006210371211532926;
              }
            } else {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.802901029586792436) ) ) {
                  result[0] += 0.009608837530509782;
                } else {
                  result[0] += -0.03694060283250233;
                }
              } else {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.44140100479126021) ) ) {
                    result[0] += 0.018614673296326694;
                  } else {
                    result[0] += -0.01286944453382656;
                  }
                } else {
                  if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += 0.02720690322074201;
                  } else {
                    result[0] += 0.0029956939485178175;
                  }
                }
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.16594791412353693) ) ) {
        result[0] += 2.032797562884699e-05;
      } else {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
          result[0] += -0.0019610551541306416;
        } else {
          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.497866153717041238) ) ) {
            if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += -0.03788451839668158;
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.071567356586456743) ) ) {
                  result[0] += -0.05593625050593518;
                } else {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += 0.03791186811060576;
                  } else {
                    result[0] += -0.0032305258834972304;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.051747083663941318) ) ) {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.637949228286744052) ) ) {
                  result[0] += -0.023720297046781882;
                } else {
                  result[0] += 0.0065784299557265745;
                }
              } else {
                result[0] += -0.033137821972638946;
              }
            }
          } else {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += -0.012359469352053006;
            } else {
              result[0] += 0.018614746023191263;
            }
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)12.00000000000000178) ) ) {
      result[0] += 0.00297701232912956;
    } else {
      result[0] += 0.1452499103527096;
    }
  }
  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)7.971558809280396396) ) ) {
    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.962127923965454546) ) ) {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.426736354827881748) ) ) {
          result[0] += 0.0007595727623319061;
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.055311203002930576) ) ) {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)12.00000000000000178) ) ) {
              if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.010064756550067985;
              } else {
                result[0] += -0.04238157536216034;
              }
            } else {
              result[0] += 0.014711408069071816;
            }
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
              result[0] += -0.00046326657311014245;
            } else {
              result[0] += -0.06153193335607226;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.046861171722413886) ) ) {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
            result[0] += -0.016127581062705218;
          } else {
            result[0] += -0.074198733508225;
          }
        } else {
          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
            result[0] += -0.004372945993372891;
          } else {
            result[0] += 0.08917108074924765;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.18134641647339045) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.124530076980591708) ) ) {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.141444921493531162) ) ) {
            if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.182021141052246982) ) ) {
                  result[0] += -0.0007203239895896816;
                } else {
                  result[0] += 0.01517301636437251;
                }
              } else {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                    if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.941167116165162021) ) ) {
                        result[0] += 0.033124604764375856;
                      } else {
                        result[0] += 8.516513757631548e-06;
                      }
                    } else {
                      result[0] += -0.008823425469686572;
                    }
                  } else {
                    if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.046861171722413886) ) ) {
                        result[0] += 0.015522072285163481;
                      } else {
                        result[0] += 0.13554013297041964;
                      }
                    } else {
                      result[0] += -0.02155549082567421;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += 0.03696674628827592;
                  } else {
                    result[0] += -0.04618969226827532;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.861792564392090288) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.778982400894165927) ) ) {
                  result[0] += -0.0007293941739266773;
                } else {
                  result[0] += 0.013577500354951698;
                }
              } else {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                  result[0] += -0.03020297583895772;
                } else {
                  result[0] += -0.002212650810662505;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += -0.0018865166069522662;
                } else {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += 0.027949087210796187;
                  } else {
                    result[0] += 0.005979125799037997;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.513969182968140537) ) ) {
                  if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.013793299834080237;
                  } else {
                    result[0] += -0.07002200087100748;
                  }
                } else {
                  if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.509355545043946201) ) ) {
                      result[0] += -0.01924804278750332;
                    } else {
                      result[0] += 0.0022284289113057333;
                    }
                  } else {
                    result[0] += 0.013941417487848369;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.01971592665799589;
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.992907285690308505) ) ) {
                  result[0] += -0.013363524032942413;
                } else {
                  result[0] += 0.019548377911433684;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.909254074096680576) ) ) {
            if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.735185861587525302) ) ) {
              if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)2.740319490432739702) ) ) {
                  result[0] += 0.01409099622555552;
                } else {
                  result[0] += -0.008969460934182702;
                }
              } else {
                if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.58491539955139249) ) ) {
                    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.329314231872559482) ) ) {
                        result[0] += -0.015391349892227783;
                      } else {
                        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                          result[0] += -0.033256245614423495;
                        } else {
                          result[0] += -0.13146840059197698;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += -0.019361821704016763;
                      } else {
                        result[0] += 0.033624423558640364;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.941167116165162021) ) ) {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += 0.09791510717343248;
                      } else {
                        result[0] += 0.014301008600057511;
                      }
                    } else {
                      result[0] += 0.0008369816821741905;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.921924352645874468) ) ) {
                    result[0] += 0.010667199182782805;
                  } else {
                    if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.743881702423096591) ) ) {
                          result[0] += -0.004258713321745925;
                        } else {
                          result[0] += -0.025026800417441014;
                        }
                      } else {
                        result[0] += -0.04697241453307063;
                      }
                    } else {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.255632162094117099) ) ) {
                        result[0] += 0.015826426076487987;
                      } else {
                        if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
                          result[0] += -0.042085622895513465;
                        } else {
                          result[0] += -0.0053745393536252595;
                        }
                      }
                    }
                  }
                }
              }
            } else {
              result[0] += -0.022270893590589688;
            }
          } else {
            result[0] += 0.0030163466823690584;
          }
        }
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.174569487571716753) ) ) {
          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += -0.00012132441887926274;
          } else {
            if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
              if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.011493357649827654;
              } else {
                if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.04760557150761253;
                } else {
                  result[0] += 0.016634815972556245;
                }
              }
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.835998296737671787) ) ) {
                result[0] += -0.022268805878552577;
              } else {
                result[0] += -0.005144847640425849;
              }
            }
          }
        } else {
          result[0] += 0.0049784681219563936;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)12.00000000000000178) ) ) {
      result[0] += 0.0022702919669489365;
    } else {
      result[0] += 0.14643189317171446;
    }
  }
  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)7.971558809280396396) ) ) {
    if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
      result[0] += -0.0005511859911957997;
    } else {
      if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += 0.0007361848128143605;
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += 0.052683404575394055;
              } else {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.422362327575684482) ) ) {
                  result[0] += -0.0317851199363138;
                } else {
                  result[0] += -0.11278342111415177;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
                if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                  result[0] += -0.006665167417934607;
                } else {
                  if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.04325255055726773;
                  } else {
                    result[0] += -0.007341762105027877;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.241523027420044833) ) ) {
                  result[0] += -0.0020871349831270373;
                } else {
                  result[0] += -0.08426368414533918;
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.97887301445007413) ) ) {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.807895898818970615) ) ) {
                      result[0] += -0.023164206295553033;
                    } else {
                      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.597323656082154208) ) ) {
                          if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
                            result[0] += -0.06911128834156886;
                          } else {
                            result[0] += 0.010677603106916704;
                          }
                        } else {
                          result[0] += -0.11261330391457441;
                        }
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.605120182037354404) ) ) {
                          result[0] += -0.025195092970860103;
                        } else {
                          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.500490188598633701) ) ) {
                            result[0] += 0.010331086026085564;
                          } else {
                            result[0] += 0.0692833632013599;
                          }
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.745876312255860263) ) ) {
                        result[0] += 0.11536442376114382;
                      } else {
                        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                          result[0] += -0.07148289017014173;
                        } else {
                          result[0] += 0.037464673276755504;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.422362327575684482) ) ) {
                        result[0] += -0.02379881296827999;
                      } else {
                        result[0] += 0.03817135347755368;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
                    result[0] += -0.004393648187343293;
                  } else {
                    result[0] += 0.01607311512546608;
                  }
                }
              } else {
                if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                    if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                      if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.436733961105347568) ) ) {
                        result[0] += 0.010479289074046903;
                      } else {
                        result[0] += -0.01562932180510176;
                      }
                    } else {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.321723937988282138) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.126503944396974433) ) ) {
                          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.591613531112671787) ) ) {
                            result[0] += 0.03956765779944647;
                          } else {
                            result[0] += -0.01905015449282853;
                          }
                        } else {
                          result[0] += -0.025532426647230495;
                        }
                      } else {
                        result[0] += -0.1027435441163034;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.88435244560241788) ) ) {
                      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.439304351806642401) ) ) {
                            result[0] += 0.006823405397075513;
                          } else {
                            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.636499762535095659) ) ) {
                              result[0] += -0.0895378521294028;
                            } else {
                              result[0] += 0.0400822734626079;
                            }
                          }
                        } else {
                          result[0] += 0.0030265279028060302;
                        }
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.636499762535095659) ) ) {
                          result[0] += 0.07201421994555555;
                        } else {
                          result[0] += -0.010631396859385982;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                        result[0] += 0.08346733857631768;
                      } else {
                        result[0] += 0.02402036576465688;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.78508520126342951) ) ) {
                    if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.51517200469970881) ) ) {
                          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.344550132751465732) ) ) {
                            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.659039497375490058) ) ) {
                              if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                                result[0] += 0.020506134719314006;
                              } else {
                                result[0] += -0.13265819943932947;
                              }
                            } else {
                              result[0] += 0.01739196964769617;
                            }
                          } else {
                            result[0] += 0.03230286957310435;
                          }
                        } else {
                          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.778982400894165927) ) ) {
                            result[0] += 0.03571248635718685;
                          } else {
                            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                              result[0] += -0.11002957326185377;
                            } else {
                              result[0] += 0.030948917250096915;
                            }
                          }
                        }
                      } else {
                        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.12508964538574396) ) ) {
                          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.623839378356934482) ) ) {
                            result[0] += -0.0012293533615805497;
                          } else {
                            result[0] += -0.045835517104931535;
                          }
                        } else {
                          result[0] += 0.031265550749681424;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                        if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.012675821781158891) ) ) {
                          result[0] += -0.09350192306356853;
                        } else {
                          if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
                            if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.735185861587525302) ) ) {
                              result[0] += -0.041970803482825614;
                            } else {
                              result[0] += 0.013773587329041014;
                            }
                          } else {
                            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.384246587753296343) ) ) {
                                result[0] += 0.025518978275519286;
                              } else {
                                result[0] += 0.12573078907083027;
                              }
                            } else {
                              result[0] += -0.0002686355026072911;
                            }
                          }
                        }
                      } else {
                        result[0] += -0.06387804208408975;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.431901693344116655) ) ) {
                      result[0] += 0.026917557483094025;
                    } else {
                      result[0] += -0.06439148655114771;
                    }
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.795762062072754794) ) ) {
            result[0] += 0.0020224722237468075;
          } else {
            result[0] += -0.03509925907765272;
          }
        }
      } else {
        result[0] += -0.00016383948401481785;
      }
    }
  } else {
    if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
      result[0] += -4.830792372291616e-05;
    } else {
      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
        result[0] += 0.19567607649907726;
      } else {
        result[0] += -0.011419820461118647;
      }
    }
  }
  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.493027687072754794) ) ) {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
          if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.004351471382911329;
            } else {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.914472818374634233) ) ) {
                      result[0] += 0.005370591146984354;
                    } else {
                      result[0] += 0.07179843246438625;
                    }
                  } else {
                    result[0] += -0.011150753715810523;
                  }
                } else {
                  if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.025192260742188388) ) ) {
                      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += -0.1516450798819417;
                      } else {
                        result[0] += -0.0417832116423606;
                      }
                    } else {
                      result[0] += -0.00886466273447299;
                    }
                  } else {
                    if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                        result[0] += 0.002914702658788257;
                      } else {
                        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += -0.03529353920290983;
                        } else {
                          result[0] += -0.004071239208803537;
                        }
                      }
                    } else {
                      result[0] += 0.011283515868381715;
                    }
                  }
                }
              } else {
                result[0] += 0.0054041721992724315;
              }
            }
          } else {
            result[0] += -0.008523618726502648;
          }
        } else {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.636499762535095659) ) ) {
            result[0] += -0.02811568180320383;
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
              result[0] += 0.009825925353396629;
            } else {
              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += 0.0027645781178641947;
                } else {
                  result[0] += -0.021811950816333164;
                }
              } else {
                if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.970085620880127397) ) ) {
                  result[0] += -0.008632912838971381;
                } else {
                  result[0] += -0.03008665336870608;
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.830334186553955966) ) ) {
            if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += 0.0022376139153478984;
              } else {
                result[0] += -0.04231662767783646;
              }
            } else {
              result[0] += -0.03702904808547034;
            }
          } else {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += -0.0066394913285444685;
            } else {
              if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2252.000000000000455) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.567862033843995029) ) ) {
                  result[0] += 0.15458951957715222;
                } else {
                  result[0] += -0.0469306625551509;
                }
              } else {
                result[0] += 0.0038050849212881276;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.993164777755738193) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.542080402374269354) ) ) {
              result[0] += 0.005833621674802041;
            } else {
              result[0] += -0.005514065981082694;
            }
          } else {
            result[0] += -0.00775306848471235;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
        if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.740319490432739702) ) ) {
              result[0] += 0.018513689980070407;
            } else {
              result[0] += -0.00257361573053252;
            }
          } else {
            if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
              if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
                if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += 0.005412820201821474;
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.770631790161133257) ) ) {
                    result[0] += 0.013872855104105037;
                  } else {
                    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
                        result[0] += -0.023883427506689793;
                      } else {
                        result[0] += -0.004950292965510004;
                      }
                    } else {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.54220247268676935) ) ) {
                        if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                          if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.494223117828370029) ) ) {
                            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.255632162094117099) ) ) {
                              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.750972747802735263) ) ) {
                                  if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)0.8958797454833985485) ) ) {
                                    result[0] += -0.08758123202515466;
                                  } else {
                                    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                                      result[0] += 0.8453376951609807;
                                    } else {
                                      result[0] += 0.1567648014409124;
                                    }
                                  }
                                } else {
                                  result[0] += -0.14033101573141823;
                                }
                              } else {
                                if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                                  result[0] += -0.10986146249915783;
                                } else {
                                  result[0] += 0.06509882037533456;
                                }
                              }
                            } else {
                              if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.417592287063599077) ) ) {
                                result[0] += -0.06029224874013054;
                              } else {
                                result[0] += 0.0828100718493402;
                              }
                            }
                          } else {
                            result[0] += 0.0395367498296299;
                          }
                        } else {
                          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.465643882751465732) ) ) {
                            result[0] += -0.09389566011962726;
                          } else {
                            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.16594791412353693) ) ) {
                              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.012675821781158891) ) ) {
                                result[0] += 0.07205619108848751;
                              } else {
                                result[0] += -0.03157557721541739;
                              }
                            } else {
                              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.81940793991089045) ) ) {
                                result[0] += 0.07191096987358572;
                              } else {
                                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                                  result[0] += 0.08923317879143519;
                                } else {
                                  result[0] += -0.032018930184510824;
                                }
                              }
                            }
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.397998809814454013) ) ) {
                          result[0] += -0.024592392927201803;
                        } else {
                          result[0] += 0.03666160139693054;
                        }
                      }
                    }
                  }
                }
              } else {
                result[0] += 0.002458854800507731;
              }
            } else {
              result[0] += 0.05920387717136158;
            }
          }
        } else {
          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.700598716735840066) ) ) {
              result[0] += -0.01432585522308762;
            } else {
              result[0] += 0.015819112180785154;
            }
          } else {
            if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
              result[0] += -0.0007803957279997173;
            } else {
              result[0] += -0.0437940735147955;
            }
          }
        }
      } else {
        result[0] += 0.0015242482365081065;
      }
    }
  } else {
    if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
        result[0] += 0.0030561080478837366;
      } else {
        if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += -0.043885815954563145;
        } else {
          result[0] += -0.0029652665195606027;
        }
      }
    } else {
      result[0] += -0.00018163565312587657;
    }
  }
  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)7.971558809280396396) ) ) {
    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.962127923965454546) ) ) {
        result[0] += 0.0005152089374202527;
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.046861171722413886) ) ) {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
            result[0] += -0.016113321843812;
          } else {
            result[0] += -0.0711416752245521;
          }
        } else {
          result[0] += -0.003649218561508475;
        }
      }
    } else {
      if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.384830474853516513) ) ) {
          if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)2.764714598655701128) ) ) {
            if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.350257158279419833) ) ) {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.825422286987305576) ) ) {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.524927973747253862) ) ) {
                  if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.00011308860568570686;
                  } else {
                    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.97887301445007413) ) ) {
                        if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                          if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                            result[0] += -0.021321854774109052;
                          } else {
                            result[0] += 0.058429800695144765;
                          }
                        } else {
                          result[0] += 0.03418377016169962;
                        }
                      } else {
                        result[0] += -0.021197548254520165;
                      }
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)2.138333082199097124) ) ) {
                        result[0] += -0.010731631168808669;
                      } else {
                        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                            if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.998158693313599077) ) ) {
                              result[0] += 0.01623431851977606;
                            } else {
                              result[0] += 0.05661148275719054;
                            }
                          } else {
                            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                              result[0] += -0.0027839189377158752;
                            } else {
                              result[0] += 0.019840482443535205;
                            }
                          }
                        } else {
                          if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                              result[0] += 0.0519796540530017;
                            } else {
                              if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.067782521247864214) ) ) {
                                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.433569431304932529) ) ) {
                                  result[0] += 0.06173788466608069;
                                } else {
                                  result[0] += -0.0008253657414802936;
                                }
                              } else {
                                result[0] += -0.034497708604183455;
                              }
                            }
                          } else {
                            if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                                result[0] += 0.02856064126822099;
                              } else {
                                result[0] += 0.1235027185479279;
                              }
                            } else {
                              result[0] += 0.028530511095564162;
                            }
                          }
                        }
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                      result[0] += -0.0062877109464966295;
                    } else {
                      result[0] += 0.0036447272396097873;
                    }
                  } else {
                    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                        if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                          result[0] += 0.024125090939279958;
                        } else {
                          result[0] += -0.006730524657168363;
                        }
                      } else {
                        result[0] += -0.02994619983419037;
                      }
                    } else {
                      result[0] += -0.036121977650112415;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.020127415657043901) ) ) {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.855921268463135654) ) ) {
                    result[0] += -0.0021464676041271246;
                  } else {
                    result[0] += -0.030134213566021764;
                  }
                } else {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                    if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                      result[0] += -0.005906677066348971;
                    } else {
                      result[0] += -0.038791068005629335;
                    }
                  } else {
                    result[0] += -0.05454370417140774;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)2.012675821781158891) ) ) {
                if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.005294604847867569;
                  } else {
                    result[0] += 0.05304489807040324;
                  }
                } else {
                  result[0] += -0.01765122366066311;
                }
              } else {
                result[0] += 0.008910325606650627;
              }
            }
          } else {
            result[0] += 0.07964158663951043;
          }
        } else {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.42478513717651456) ) ) {
            result[0] += 0.032590297699565965;
          } else {
            if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.126931190490723544) ) ) {
              if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.556798219680787021) ) ) {
                    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                      if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                        result[0] += 0.03270844737022741;
                      } else {
                        result[0] += -0.08279831172300663;
                      }
                    } else {
                      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.868834793567657693) ) ) {
                        result[0] += -0.042242797216912965;
                      } else {
                        result[0] += -0.005560116613332204;
                      }
                    }
                  } else {
                    result[0] += 0.0009674122633261007;
                  }
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.714269638061524326) ) ) {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                      if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.861792564392090288) ) ) {
                        result[0] += 0.010289337316469648;
                      } else {
                        result[0] += -0.01450448521642517;
                      }
                    } else {
                      result[0] += 0.019973955090275625;
                    }
                  } else {
                    result[0] += 0.05473854750263102;
                  }
                }
              } else {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                  if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                    if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += -0.06138421629972693;
                    } else {
                      result[0] += 0.06838968240411064;
                    }
                  } else {
                    if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.740319490432739702) ) ) {
                      result[0] += 0.01079449899752287;
                    } else {
                      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.367881059646607333) ) ) {
                          result[0] += -0.02082824233198185;
                        } else {
                          result[0] += 0.012993684826350963;
                        }
                      } else {
                        result[0] += -0.061142508014059786;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.766185760498047763) ) ) {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += 0.04461520896566711;
                    } else {
                      result[0] += -0.012964766269843218;
                    }
                  } else {
                    result[0] += 0.05417331752078878;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += -0.0028217523383657277;
              } else {
                result[0] += -0.030897753015367485;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.97887301445007413) ) ) {
          result[0] += -0.00025573285450620234;
        } else {
          result[0] += -0.0227192023371978;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
      result[0] += -6.424017693563584e-05;
    } else {
      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
        result[0] += 0.1957220308031581;
      } else {
        result[0] += -0.009445449013657346;
      }
    }
  }
  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.493027687072754794) ) ) {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.744781017303467685) ) ) {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.510617971420288974) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.938867926597595659) ) ) {
                  if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)2.500000000000000444) ) ) {
                    result[0] += -0.10489007830104294;
                  } else {
                    result[0] += -0.010624092080089744;
                  }
                } else {
                  if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += -0.007314079088799982;
                  } else {
                    if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += 0.004764722394015242;
                    } else {
                      result[0] += 0.0353598645780885;
                    }
                  }
                }
              } else {
                result[0] += -0.014328578099138116;
              }
            } else {
              result[0] += 0.003662819734090368;
            }
          } else {
            if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.700598716735840066) ) ) {
              result[0] += 0.0028794689895808127;
            } else {
              result[0] += -0.06806711869186499;
            }
          }
        } else {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.636499762535095659) ) ) {
            result[0] += -0.027121244586638272;
          } else {
            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.970085620880127397) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.924581527709961826) ) ) {
                result[0] += 0.00469971845714981;
              } else {
                result[0] += -0.0168153904057076;
              }
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.029068946838379794) ) ) {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += -0.016554750471756;
                  } else {
                    if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.04033867669803204;
                    } else {
                      result[0] += 0.004889520444027006;
                    }
                  }
                } else {
                  result[0] += -0.018084510084618848;
                }
              } else {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                  result[0] += -0.01272258230311018;
                } else {
                  result[0] += -0.041520350424150344;
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.020127415657043901) ) ) {
          if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
              if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += 0.0050207460482004996;
                } else {
                  result[0] += -0.02868964338110573;
                }
              } else {
                result[0] += 0.01718285875269737;
              }
            } else {
              result[0] += -0.004221800089203781;
            }
          } else {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.56941866874694913) ) ) {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.012299312743825438;
                } else {
                  result[0] += 0.002048965661349925;
                }
              } else {
                result[0] += -0.0472935397597662;
              }
            } else {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += 0.009018498474229631;
              } else {
                result[0] += -0.017313440824226277;
              }
            }
          }
        } else {
          result[0] += -0.005798700635428112;
        }
      }
    } else {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
        if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.740319490432739702) ) ) {
              result[0] += 0.017942122012085716;
            } else {
              result[0] += -0.002491341575032065;
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.921060562133789951) ) ) {
              if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.005290634597760372;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)2.740319490432739702) ) ) {
                  result[0] += 0.023103964439869354;
                } else {
                  if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.802696108818054643) ) ) {
                      result[0] += -0.00786200621061163;
                    } else {
                      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += -0.063098151842625;
                      } else {
                        result[0] += -0.013842894914819806;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.941167116165162021) ) ) {
                      result[0] += -0.07076607204132578;
                    } else {
                      result[0] += 0.006451482108105886;
                    }
                  }
                }
              }
            } else {
              result[0] += -0.002333227467199456;
            }
          }
        } else {
          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
            result[0] += 0.010410701786224663;
          } else {
            if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
              result[0] += 0.052302937193697024;
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.154959201812744585) ) ) {
                if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.04420640737740267;
                } else {
                  result[0] += 0.011238658417317547;
                }
              } else {
                result[0] += -0.0008231413716557035;
              }
            }
          }
        }
      } else {
        result[0] += 0.0013995419617844251;
      }
    }
  } else {
    if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
        if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
          result[0] += 0.003278576588632603;
        } else {
          if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.88435244560241788) ) ) {
              if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += -0.09576571651292641;
                } else {
                  result[0] += -0.011628334491868968;
                }
              } else {
                result[0] += 0.004426461491567552;
              }
            } else {
              result[0] += 0.02925132172049823;
            }
          } else {
            if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.770631790161133257) ) ) {
                result[0] += 0.13207993195991471;
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.289595603942871982) ) ) {
                  result[0] += -0.01719036196782053;
                } else {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += -0.07865803033204649;
                  } else {
                    result[0] += 0.032237858632308464;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.51918649673462092) ) ) {
                result[0] += 0.009923765400147042;
              } else {
                if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += 0.02004016138415332;
                } else {
                  result[0] += -0.06953598310733274;
                }
              }
            }
          }
        }
      } else {
        result[0] += -0.022882013001752494;
      }
    } else {
      if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.363078355789185458) ) ) {
        result[0] += -0.0004464016156957061;
      } else {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
          result[0] += 0.010384407744801032;
        } else {
          if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
              result[0] += -0.05033243525091167;
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.53326439857482999) ) ) {
                result[0] += 0.04359437184223247;
              } else {
                result[0] += -0.0036124028728814588;
              }
            }
          } else {
            result[0] += -0.029008310765406525;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)2.764714598655701128) ) ) {
    if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
      if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.993164777755738193) ) ) {
          if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += -0.021152429603274805;
          } else {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.524927973747253862) ) ) {
              result[0] += 0.03461675468651983;
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
                result[0] += 0.03813740226219073;
              } else {
                result[0] += -0.02441306240601457;
              }
            }
          }
        } else {
          result[0] += -0.061864599882201214;
        }
      } else {
        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.182021141052246982) ) ) {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.872538805007935458) ) ) {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.310776710510254794) ) ) {
                  result[0] += 0.0009008987701973946;
                } else {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.835998296737671787) ) ) {
                      result[0] += 0.0028938819212532634;
                    } else {
                      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                        result[0] += -0.0017763225644213847;
                      } else {
                        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.962127923965454546) ) ) {
                          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.397998809814454013) ) ) {
                            result[0] += 0.0016423701387574985;
                          } else {
                            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.909855604171753818) ) ) {
                                result[0] += -0.017122419781056067;
                              } else {
                                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                                  result[0] += -0.04209008555804794;
                                } else {
                                  result[0] += -0.2291505000454163;
                                }
                              }
                            } else {
                              result[0] += 0.0030070373757977104;
                            }
                          }
                        } else {
                          result[0] += -0.07840996096521069;
                        }
                      }
                    }
                  } else {
                    result[0] += 0.04398465581242114;
                  }
                }
              } else {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += 0.005711277804748707;
                } else {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.511434078216553178) ) ) {
                    if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                      result[0] += -0.004214646905585272;
                    } else {
                      result[0] += -0.04454081535377436;
                    }
                  } else {
                    result[0] += 0.023125838061650267;
                  }
                }
              }
            } else {
              result[0] += -0.03399135008504801;
            }
          } else {
            if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.740319490432739702) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.714269638061524326) ) ) {
                result[0] += 0.00023714168370998672;
              } else {
                result[0] += 0.07773474848783879;
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.241523027420044833) ) ) {
                result[0] += 0.0016603998990550749;
              } else {
                result[0] += -0.02191440136583148;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.158952236175537998) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
                result[0] += -0.002420762398691473;
              } else {
                result[0] += -0.04190192790238158;
              }
            } else {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.450390577316285068) ) ) {
                  result[0] += -0.004279292796486587;
                } else {
                  result[0] += -0.01800903335380506;
                }
              } else {
                result[0] += -0.03665051706693625;
              }
            }
          } else {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2252.000000000000455) ) ) {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.620046615600586826) ) ) {
                result[0] += -0.2381636767835178;
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.795762062072754794) ) ) {
                  result[0] += -0.10153368959206376;
                } else {
                  result[0] += 0.010431051948396117;
                }
              }
            } else {
              result[0] += 0.00768869148057039;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.131699204444885698) ) ) {
        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.770631790161133257) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.761470437049866167) ) ) {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
              result[0] += 0.005182434223087484;
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.801954269409180576) ) ) {
                result[0] += -0.03311276414728156;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.142630577087403232) ) ) {
                  result[0] += 0.07855474585528763;
                } else {
                  result[0] += -0.009137833325186378;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += -0.02269436130256325;
            } else {
              result[0] += 0.022313870284065552;
            }
          }
        } else {
          result[0] += -0.014107338963459767;
        }
      } else {
        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.212100267410279208) ) ) {
          if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                  result[0] += -0.09146406915762273;
                } else {
                  result[0] += -0.016612568588091795;
                }
              } else {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += -0.01672786095515297;
                } else {
                  if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += 0.007603463706277912;
                  } else {
                    result[0] += 0.0289987960330801;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.071567356586456743) ) ) {
                result[0] += 9.567078148387416e-05;
              } else {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += 0.018529886510117152;
                } else {
                  result[0] += -0.023189534961730332;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.182065486907959873) ) ) {
                if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.00816992100150431;
                } else {
                  result[0] += -0.03954848671038888;
                }
              } else {
                result[0] += -0.05346021493682197;
              }
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.384830474853516513) ) ) {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                  if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.58961367607116788) ) ) {
                    result[0] += -0.020612131020432453;
                  } else {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.156774044036865678) ) ) {
                      result[0] += -0.010199771391421808;
                    } else {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.745876312255860263) ) ) {
                        if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.624251961708069292) ) ) {
                          result[0] += -0.023980871826914968;
                        } else {
                          result[0] += 0.020962848730209746;
                        }
                      } else {
                        result[0] += 0.05620000125312335;
                      }
                    }
                  }
                } else {
                  result[0] += 0.02078377187392512;
                }
              } else {
                result[0] += 0.0060036598401007925;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
            result[0] += 0.0028749920759532175;
          } else {
            if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.505036115646363193) ) ) {
              result[0] += 0.006724875232376103;
            } else {
              result[0] += 0.039365183586152236;
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
      result[0] += -0.04506400818973723;
    } else {
      result[0] += 0.041023155287361536;
    }
  }
  if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)2.764714598655701128) ) ) {
    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.102759599685669833) ) ) {
          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.835998296737671787) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.795762062072754794) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.441542863845826083) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.91907978057861506) ) ) {
                    result[0] += 0.07634789223756495;
                  } else {
                    result[0] += -0.0044307466869930355;
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.662244915962219682) ) ) {
                    result[0] += 0.019733533432774994;
                  } else {
                    result[0] += -0.04345138213819907;
                  }
                }
              } else {
                result[0] += -0.13878329207818016;
              }
            } else {
              result[0] += 0.003970297607623268;
            }
          } else {
            result[0] += 0.019151138345087262;
          }
        } else {
          result[0] += 0.07508146929048598;
        }
      } else {
        if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
              result[0] += 0.0029773965314501788;
            } else {
              result[0] += -0.005977939804550843;
            }
          } else {
            result[0] += -0.0018411284862847818;
          }
        } else {
          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)6.000000000000000888) ) ) {
              if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                result[0] += 0.014843598982333468;
              } else {
                if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.0033635592680975445;
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.55604696273803889) ) ) {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.876230478286744052) ) ) {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.51918649673462092) ) ) {
                        result[0] += -0.0020997807868926572;
                      } else {
                        result[0] += -0.2183746975914727;
                      }
                    } else {
                      result[0] += 0.023256791256626802;
                    }
                  } else {
                    result[0] += 0.03623611886519003;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += -0.00174446015690332;
              } else {
                if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.008959688627355807;
                  } else {
                    if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.017515603801866432;
                    } else {
                      result[0] += -0.023949037509619343;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.982408046722412998) ) ) {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.35306882858276456) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.241523027420044833) ) ) {
                        result[0] += 0.02286970375215699;
                      } else {
                        result[0] += -0.006065192344022606;
                      }
                    } else {
                      result[0] += -0.042989981403218976;
                    }
                  } else {
                    result[0] += -0.030141768668335612;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.467917680740357333) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
                    result[0] += 0.05684035796020384;
                  } else {
                    result[0] += -0.03505844911943281;
                  }
                } else {
                  result[0] += -0.09301930911126009;
                }
              } else {
                result[0] += -0.0004828570069439098;
              }
            } else {
              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.216319084167481357) ) ) {
                    result[0] += 0.04057059277714789;
                  } else {
                    result[0] += 0.11169823323692772;
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.940167903900147373) ) ) {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                      if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)2.012675821781158891) ) ) {
                        if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += 0.00030877992872727295;
                        } else {
                          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                            result[0] += 0.020642853915182658;
                          } else {
                            result[0] += -0.024318055550303155;
                          }
                        }
                      } else {
                        result[0] += -0.14528702187927992;
                      }
                    } else {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.993164777755738193) ) ) {
                        result[0] += -0.003797004310927946;
                      } else {
                        result[0] += 0.030274905658195156;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                        result[0] += -0.005726110177126434;
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.625595092773438388) ) ) {
                          result[0] += -0.003791544433212786;
                        } else {
                          result[0] += 0.012406595500590126;
                        }
                      }
                    } else {
                      result[0] += 0.013550758439380847;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[10].missing != -1) || (data[10].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
                    result[0] += 0.0008409149122255138;
                  } else {
                    result[0] += -0.01058051396223414;
                  }
                } else {
                  result[0] += -0.018183121622352184;
                }
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.637949228286744052) ) ) {
          if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += -0.009514183715655407;
          } else {
            result[0] += -0.052424616088075096;
          }
        } else {
          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)2.012675821781158891) ) ) {
            result[0] += -0.07034565409467539;
          } else {
            result[0] += 0.08772988465664461;
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)2.138333082199097124) ) ) {
          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.025192260742188388) ) ) {
              result[0] += 0.08586355286564336;
            } else {
              result[0] += 0.0020373724978861156;
            }
          } else {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += 0.13734378590008872;
            } else {
              result[0] += 0.003839727017438336;
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.439304351806642401) ) ) {
            if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)2.012675821781158891) ) ) {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.397998809814454013) ) ) {
                  result[0] += 0.03133002359092898;
                } else {
                  if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.019220249620373598;
                  } else {
                    result[0] += -0.059539653300895314;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += 0.025817272853799436;
                } else {
                  result[0] += -0.014710247095115928;
                }
              }
            } else {
              result[0] += 0.0549425822120056;
            }
          } else {
            result[0] += -0.07464334426290685;
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
      if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
        if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
          result[0] += -0.022357730507640096;
        } else {
          result[0] += -0.07781013501957149;
        }
      } else {
        result[0] += -0.006813112399808215;
      }
    } else {
      result[0] += 0.045232785491423;
    }
  }
  if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)2.764714598655701128) ) ) {
    if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
      if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.674522399902344638) ) ) {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.334978580474854404) ) ) {
            result[0] += 0.021319314531440908;
          } else {
            result[0] += -0.07859590823116447;
          }
        } else {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.342454433441162998) ) ) {
            result[0] += 0.0004606190272888048;
          } else {
            result[0] += -0.04560851338574806;
          }
        }
      } else {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.91907978057861506) ) ) {
          result[0] += 0.0002753220328770268;
        } else {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.888826131820679155) ) ) {
            result[0] += 0.007363075088072237;
          } else {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
              result[0] += -0.0010083558559883352;
            } else {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.384830474853516513) ) ) {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += -0.004231370555047109;
                } else {
                  result[0] += -0.01957907076072304;
                }
              } else {
                result[0] += -0.053710003070557455;
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
          if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.594915628433228427) ) ) {
              if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.505036115646363193) ) ) {
                result[0] += -0.03619225753571179;
              } else {
                result[0] += 0.010365036066630647;
              }
            } else {
              result[0] += 0.017095596692760864;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.664408206939698154) ) ) {
              result[0] += -0.01022867687413266;
            } else {
              if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.0001168633367220039;
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.247576236724854404) ) ) {
                  result[0] += -0.012653009591191948;
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.439304351806642401) ) ) {
                    result[0] += 0.009690863444602772;
                  } else {
                    result[0] += 0.0392966122441861;
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
            if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
              result[0] += -0.0024744502585250646;
            } else {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.553712725639343706) ) ) {
                  result[0] += 0.02860859641091448;
                } else {
                  result[0] += -0.047822295639674973;
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.596743106842042792) ) ) {
                  result[0] += -0.038771870114869676;
                } else {
                  if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.851041555404663974) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.24222278594970881) ) ) {
                      result[0] += -0.02862850017353439;
                    } else {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.262283086776734287) ) ) {
                        result[0] += -0.042479064780371285;
                      } else {
                        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                          if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            result[0] += -0.017517899664933473;
                          } else {
                            result[0] += 0.047710962815493196;
                          }
                        } else {
                          result[0] += 0.07628530475909855;
                        }
                      }
                    }
                  } else {
                    result[0] += 0.04823669849706959;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.357691764831543413) ) ) {
              result[0] += 0.06317325794431632;
            } else {
              if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.434600353240968573) ) ) {
                  result[0] += -0.051652712658149594;
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.467917680740357333) ) ) {
                    result[0] += -0.0686555226414642;
                  } else {
                    result[0] += 0.06490260823693582;
                  }
                }
              } else {
                result[0] += -0.0626261022656229;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
              result[0] += 0.0629517265229676;
            } else {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.509355545043946201) ) ) {
                result[0] += -0.03144720740500612;
              } else {
                result[0] += -0.08589528529893821;
              }
            }
          } else {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.917405366897583452) ) ) {
              if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += -0.028777558497486033;
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.5240359306335467) ) ) {
                  result[0] += -0.0054779117165611665;
                } else {
                  result[0] += 0.09369108520789649;
                }
              }
            } else {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += 0.01803840261720085;
              } else {
                result[0] += -0.01119142549764367;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.216319084167481357) ) ) {
                result[0] += 0.035844481858458614;
              } else {
                result[0] += 0.10061861177398489;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.605120182037354404) ) ) {
                result[0] += -0.007166022843164058;
              } else {
                if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.321723937988282138) ) ) {
                    if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.0027773647855887196;
                    } else {
                      result[0] += 0.006134065649364284;
                    }
                  } else {
                    if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += -0.024422832688203486;
                    } else {
                      if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += 0.042675622494307605;
                      } else {
                        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                          result[0] += 0.036544218767044355;
                        } else {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.318498134613038886) ) ) {
                            result[0] += 0.02049521116625656;
                          } else {
                            result[0] += -0.07973152594504802;
                          }
                        }
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.02510868565827175;
                  } else {
                    result[0] += 0.008736825890965182;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
              if ( LIKELY( !(data[10].missing != -1) || (data[10].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
                  result[0] += 0.00041475731078326856;
                } else {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.985194206237793857) ) ) {
                    result[0] += -0.013204203380187652;
                  } else {
                    result[0] += 0.027019072964349324;
                  }
                }
              } else {
                result[0] += -0.01813270490350848;
              }
            } else {
              result[0] += 0.011471304973176825;
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
        result[0] += -0.06016673096868043;
      } else {
        result[0] += 0.0417340437635178;
      }
    } else {
      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
        if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
          result[0] += -0.06317922017450316;
        } else {
          result[0] += 0.05288499919695984;
        }
      } else {
        result[0] += -0.08636682759767554;
      }
    }
  }
  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.126503944396974433) ) ) {
      if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
          if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)4.125962495803833896) ) ) {
            result[0] += 6.42859791886386e-05;
          } else {
            result[0] += 0.0573222338205388;
          }
        } else {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.837713479995728427) ) ) {
            result[0] += -0.003781463099696124;
          } else {
            if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += -0.009115987793931279;
            } else {
              result[0] += -0.056292938518872696;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
          result[0] += -0.05245960246996004;
        } else {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.334978580474854404) ) ) {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.909855604171753818) ) ) {
              result[0] += -0.01558851167294394;
            } else {
              if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.26188084778162385;
              } else {
                result[0] += -0.06163721587325153;
              }
            }
          } else {
            result[0] += 0.026083145475342386;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.400584220886231357) ) ) {
          result[0] += 0.003612418313424299;
        } else {
          result[0] += -0.008777788705065667;
        }
      } else {
        if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
          if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)12.00000000000000178) ) ) {
            result[0] += 0.00045648950640307727;
          } else {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.088880300521851474) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                result[0] += 0.00530563752464842;
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.142630577087403232) ) ) {
                  if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)2.012675821781158891) ) ) {
                    if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5.059420347213746005) ) ) {
                      result[0] += -0.0016010571587043517;
                    } else {
                      result[0] += 0.11642230422381944;
                    }
                  } else {
                    result[0] += -0.046861293225006095;
                  }
                } else {
                  result[0] += -0.05938250258879341;
                }
              }
            } else {
              result[0] += -0.06110277353833502;
            }
          }
        } else {
          result[0] += 0.002973330499692917;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.16594791412353693) ) ) {
      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.23832273483276456) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.105651378631592685) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.766185760498047763) ) ) {
              result[0] += 0.013440497732362484;
            } else {
              result[0] += -0.010570775475707415;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)2.138333082199097124) ) ) {
              result[0] += 0.024796406898611154;
            } else {
              result[0] += -0.02133925401333915;
            }
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.531673669815064365) ) ) {
            result[0] += 0.16229887221038777;
          } else {
            if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.288254261016846591) ) ) {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += 0.04021397226527637;
                } else {
                  result[0] += -0.01192277058312409;
                }
              } else {
                result[0] += -0.010980244803956509;
              }
            } else {
              result[0] += 0.01985704946524723;
            }
          }
        }
      } else {
        result[0] += 0.0003025016869184373;
      }
    } else {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
        if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
            result[0] += -0.033258256407202494;
          } else {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
              result[0] += 0.006038257881527972;
            } else {
              if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                result[0] += -0.07058951557541411;
              } else {
                result[0] += -0.00497054092745524;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.321723937988282138) ) ) {
            if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.938867926597595659) ) ) {
                result[0] += 0.06798708354718737;
              } else {
                result[0] += -0.006265490046849948;
              }
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.993164777755738193) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                  result[0] += -0.02077414395592062;
                } else {
                  result[0] += -0.04535066232694346;
                }
              } else {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += -0.004119219361829703;
                } else {
                  result[0] += -0.03332952748456207;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += -0.04247330816661335;
            } else {
              result[0] += 0.05970958588091189;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
            if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.923617362976075107) ) ) {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)12.00000000000000178) ) ) {
                result[0] += 0.0018068189439099025;
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.718933820724488193) ) ) {
                  result[0] += 0.034923196486577955;
                } else {
                  result[0] += -0.05494226057771713;
                }
              }
            } else {
              if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.01833468834065271;
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.302512168884278232) ) ) {
                  result[0] += -0.018444829418523056;
                } else {
                  result[0] += 0.03803569021441206;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.510617971420288974) ) ) {
              if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += 0.02202044976876344;
              } else {
                if ( UNLIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.07908024771713554;
                  } else {
                    result[0] += -0.031204444028224473;
                  }
                } else {
                  result[0] += -0.02918047805351925;
                }
              }
            } else {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.05254968408251248;
              } else {
                if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += -0.02846130705696137;
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.12772703170776545) ) ) {
                    result[0] += 0.00914426175964241;
                  } else {
                    if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += -0.015058317180031054;
                    } else {
                      result[0] += 0.04684946914579058;
                    }
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.142630577087403232) ) ) {
              result[0] += -0.016677939967416375;
            } else {
              result[0] += 0.0025780988325055836;
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += 0.020260931372634766;
              } else {
                result[0] += -0.0316611625960942;
              }
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.071567356586456743) ) ) {
                result[0] += 0.09537538114733601;
              } else {
                result[0] += -0.0015388501968257356;
              }
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.00000001800250948e-35) ) ) {
    result[0] += 0.007605210218571402;
  } else {
    if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.153024196624756748) ) ) {
        result[0] += 0.0017393987963434242;
      } else {
        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.97887301445007413) ) ) {
            result[0] += 0.01326190563505838;
          } else {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.467917680740357333) ) ) {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += 0.00016612134363598068;
              } else {
                result[0] += -0.06861233004608124;
              }
            } else {
              result[0] += -0.0655036939889558;
            }
          }
        } else {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.967588424682618964) ) ) {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.43450713157653853) ) ) {
                result[0] += 0.0003689687423698569;
              } else {
                result[0] += -0.09592432566209223;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.129040718078614169) ) ) {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.837713479995728427) ) ) {
                  result[0] += -0.007041780062536052;
                } else {
                  result[0] += 0.015996665211432722;
                }
              } else {
                result[0] += -0.021305165574111468;
              }
            }
          } else {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.493027687072754794) ) ) {
                result[0] += -0.0015267027972714865;
              } else {
                if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += -0.004609462274849979;
                } else {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.321723937988282138) ) ) {
                    if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                      result[0] += -0.010829443697890597;
                    } else {
                      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                        if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                          result[0] += -0.069537768656058;
                        } else {
                          result[0] += -0.028940280137946646;
                        }
                      } else {
                        result[0] += -0.012389229496057863;
                      }
                    }
                  } else {
                    result[0] += -0.07231846287285736;
                  }
                }
              }
            } else {
              result[0] += 0.0007221381078036392;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
          if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
            result[0] += 0.011352841020365712;
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.941167116165162021) ) ) {
              result[0] += -0.00793657673279743;
            } else {
              result[0] += 0.0036090021515986386;
            }
          }
        } else {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.293085813522339311) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
                result[0] += 0.013109030280560428;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.59565925598144709) ) ) {
                  result[0] += -0.005516095015512507;
                } else {
                  result[0] += -0.04865102163097301;
                }
              }
            } else {
              if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += -0.09789948850499858;
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.285887241363526279) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.055311203002930576) ) ) {
                    if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.070700883865357333) ) ) {
                        result[0] += -0.012939035580140652;
                      } else {
                        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                          result[0] += 0.007493565309986752;
                        } else {
                          result[0] += 0.05883775260088732;
                        }
                      }
                    } else {
                      result[0] += -0.024869820352473544;
                    }
                  } else {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.892608642578125888) ) ) {
                      result[0] += 0.0022686521014372166;
                    } else {
                      result[0] += 0.015150519068320108;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.980702877044679511) ) ) {
                    result[0] += -0.027914623026253915;
                  } else {
                    result[0] += 0.012462004109755252;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.094205617904663974) ) ) {
                    result[0] += 0.015417533714281777;
                  } else {
                    result[0] += -0.04668996478574597;
                  }
                } else {
                  result[0] += -0.00010073214977914239;
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.921924352645874468) ) ) {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.48738741874694913) ) ) {
                    result[0] += -0.0076515845166068895;
                  } else {
                    result[0] += -0.1004769543084687;
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.397998809814454013) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.784468173980714667) ) ) {
                      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                        result[0] += -0.05871745480223356;
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.055311203002930576) ) ) {
                          if ( UNLIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            result[0] += -0.10081544140655177;
                          } else {
                            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.97887301445007413) ) ) {
                              result[0] += -0.07213925479489543;
                            } else {
                              result[0] += 0.06656327488831555;
                            }
                          }
                        } else {
                          result[0] += 0.033173320651944574;
                        }
                      }
                    } else {
                      result[0] += 0.011535878636293039;
                    }
                  } else {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.249904870986938921) ) ) {
                      result[0] += 0.017073487302599014;
                    } else {
                      result[0] += -0.008833725512117115;
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.97887301445007413) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
                  result[0] += 0.04246393393208976;
                } else {
                  result[0] += 0.00033411577843353547;
                }
              } else {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.13002538681030451) ) ) {
                        result[0] += 0.059837733011031215;
                      } else {
                        result[0] += -0.02576446318825276;
                      }
                    } else {
                      if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.012675821781158891) ) ) {
                        result[0] += 0.0735888628050929;
                      } else {
                        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                          result[0] += 0.02369080424481609;
                        } else {
                          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.579273939132691318) ) ) {
                            result[0] += -0.04189649939226396;
                          } else {
                            result[0] += -0.0197052711342926;
                          }
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.778982400894165927) ) ) {
                        result[0] += -0.01990879593029972;
                      } else {
                        if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
                          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
                            result[0] += 0.014678231169991736;
                          } else {
                            result[0] += -0.06339713490545516;
                          }
                        } else {
                          result[0] += -0.049805778793573674;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.909102678298951083) ) ) {
                        result[0] += 0.004767149898142467;
                      } else {
                        result[0] += -0.030945605076064697;
                      }
                    }
                  }
                } else {
                  result[0] += -0.04155682798245772;
                }
              }
            }
          }
        }
      } else {
        result[0] += 0.0013994670436241548;
      }
    }
  }
  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
    if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.029068946838379794) ) ) {
        if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.941167116165162021) ) ) {
            result[0] += -0.00500007871139924;
          } else {
            if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.497866153717041238) ) ) {
                result[0] += -0.1412025951218498;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.029068946838379794) ) ) {
                  result[0] += 0.0019023163474642534;
                } else {
                  if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.025192260742188388) ) ) {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.823630809783937323) ) ) {
                        result[0] += 0.03419046734639366;
                      } else {
                        result[0] += -0.03260662528757389;
                      }
                    } else {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.44140100479126021) ) ) {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.397998809814454013) ) ) {
                          result[0] += -0.13769346501870888;
                        } else {
                          result[0] += -0.02665706498164576;
                        }
                      } else {
                        result[0] += 0.006504305588219847;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)4.125962495803833896) ) ) {
                      if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                        result[0] += 0.009337455594265891;
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.636499762535095659) ) ) {
                          if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.783307552337647373) ) ) {
                            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.513699531555176669) ) ) {
                              result[0] += 0.04358176460567391;
                            } else {
                              if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                                if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.881510615348816362) ) ) {
                                  result[0] += -0.0775957751784105;
                                } else {
                                  result[0] += 0.07805628759160692;
                                }
                              } else {
                                result[0] += -0.04903500805679837;
                              }
                            }
                          } else {
                            if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                              result[0] += 0.1335827240925333;
                            } else {
                              result[0] += -0.014156040183476057;
                            }
                          }
                        } else {
                          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.407877445220948154) ) ) {
                            result[0] += 0.03102811907462649;
                          } else {
                            result[0] += 0.11211165089023567;
                          }
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                        result[0] += 0.1094741822931664;
                      } else {
                        result[0] += 0.0013087746206095876;
                      }
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.071567356586456743) ) ) {
                    result[0] += -0.027380392674343758;
                  } else {
                    result[0] += 0.017817573687342558;
                  }
                } else {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
                      result[0] += 0.018640773799874142;
                    } else {
                      result[0] += -0.00299997694666913;
                    }
                  } else {
                    result[0] += -0.014608538774229393;
                  }
                }
              } else {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.060294389724732333) ) ) {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)1.242453336715698464) ) ) {
                      result[0] += 0.09556639118956337;
                    } else {
                      result[0] += -0.0959711088009525;
                    }
                  } else {
                    result[0] += -0.012229795600003882;
                  }
                } else {
                  if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += -0.021680619472431162;
                  } else {
                    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.300811052322388583) ) ) {
                        result[0] += 0.020266861679318016;
                      } else {
                        result[0] += 0.08716067606803471;
                      }
                    } else {
                      result[0] += -0.00173527920334168;
                    }
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)12.00000000000000178) ) ) {
            result[0] += 0.002868871551708646;
          } else {
            if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.020064430583417236;
              } else {
                if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.002203981535894365;
                } else {
                  result[0] += -0.015715810260096046;
                }
              }
            } else {
              if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.938867926597595659) ) ) {
                  result[0] += -0.04284273525509506;
                } else {
                  result[0] += -0.0030197913621064394;
                }
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.097527027130127841) ) ) {
                  result[0] += 0.01614043175947335;
                } else {
                  result[0] += -0.015975821942793598;
                }
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.673553824424744096) ) ) {
                if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += -0.0001914540459664535;
                } else {
                  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                      result[0] += 0.004613963680363783;
                    } else {
                      result[0] += -0.029436397913781543;
                    }
                  } else {
                    result[0] += -0.01897445388176341;
                  }
                }
              } else {
                result[0] += 0.0020109241163374903;
              }
            } else {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += -0.007684088878606306;
                } else {
                  result[0] += 0.028858267912299645;
                }
              } else {
                if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.056097030639650214) ) ) {
                    result[0] += -0.053058494736562746;
                  } else {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += -0.022130521359591364;
                    } else {
                      if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)10.80270338058471857) ) ) {
                        result[0] += 0.04521551182997498;
                      } else {
                        result[0] += -0.07945378067062996;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.49584054946899592) ) ) {
                    result[0] += -0.06979481400964171;
                  } else {
                    result[0] += 0.01405507524635408;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += 0.1768965548352104;
            } else {
              result[0] += 0.028208660798902975;
            }
          }
        } else {
          result[0] += -0.011128880236307892;
        }
      }
    } else {
      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)7.971558809280396396) ) ) {
        result[0] += 0.001999238959802215;
      } else {
        result[0] += -0.08533673612550778;
      }
    }
  } else {
    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.16594791412353693) ) ) {
      result[0] += -3.4548575537008475e-05;
    } else {
      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
        result[0] += -0.002522509060427724;
      } else {
        if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
            result[0] += -0.02028056061222586;
          } else {
            result[0] += 0.0031859491655077117;
          }
        } else {
          if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += -0.007791970460854013;
          } else {
            result[0] += -0.030694559835573804;
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
    result[0] += 0.0007838352406798935;
  } else {
    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)14.25333833694458185) ) ) {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.267844915390015537) ) ) {
        if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.37109279632568537) ) ) {
            result[0] += 0.006243785100005523;
          } else {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += -0.0033133345060408314;
            } else {
              result[0] += -0.036587832264956696;
            }
          }
        } else {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.835998296737671787) ) ) {
            result[0] += -0.0041936895337047544;
          } else {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += 0.004381250501106354;
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.82428741455078303) ) ) {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += -0.003319617171854992;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.596743106842042792) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
                        result[0] += -0.08389901272330057;
                      } else {
                        if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += -0.08236499214721403;
                        } else {
                          if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                            if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                              result[0] += -0.03203119569706095;
                            } else {
                              result[0] += 0.04022380499920003;
                            }
                          } else {
                            result[0] += -0.04602261073493074;
                          }
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += 0.024823650395659334;
                      } else {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.802696108818054643) ) ) {
                          result[0] += -0.09911775838667487;
                        } else {
                          if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.350240230560303178) ) ) {
                            result[0] += 0.05028062790344798;
                          } else {
                            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)12.00000000000000178) ) ) {
                              result[0] += -0.027657259367574023;
                            } else {
                              result[0] += 0.017082121694668034;
                            }
                          }
                        }
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.242078304290772373) ) ) {
                      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                        if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.64367699623107999) ) ) {
                          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.69067406654357999) ) ) {
                            result[0] += -0.014259118937422955;
                          } else {
                            if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                              result[0] += -0.010954604293289646;
                            } else {
                              result[0] += -0.10688140830198037;
                            }
                          }
                        } else {
                          result[0] += -0.007315397602661142;
                        }
                      } else {
                        result[0] += 0.005234887046855233;
                      }
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.909102678298951083) ) ) {
                        if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += -0.0008366098050269352;
                        } else {
                          result[0] += 0.07240207661659695;
                        }
                      } else {
                        result[0] += -0.010245129046341978;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.856657028198243964) ) ) {
                      if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.241300821304322177) ) ) {
                        if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.03420138359069913) ) ) {
                          result[0] += 0.0009735118119935997;
                        } else {
                          result[0] += 0.02699131676196914;
                        }
                      } else {
                        result[0] += -0.06005021071703018;
                      }
                    } else {
                      if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                        result[0] += 0.10185214837785778;
                      } else {
                        result[0] += 0.009627623218790671;
                      }
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
                  result[0] += 0.017867232307702503;
                } else {
                  result[0] += -0.03282932317914287;
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.513699531555176669) ) ) {
          result[0] += 0.0004572863109466522;
        } else {
          if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
            if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)6.000000000000000888) ) ) {
              if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.700598716735840066) ) ) {
                  result[0] += -0.02948616159598544;
                } else {
                  if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.700598716735840066) ) ) {
                    result[0] += -0.003999461541392859;
                  } else {
                    result[0] += -0.03901165888132997;
                  }
                }
              } else {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
                  result[0] += 0.038156794989613864;
                } else {
                  result[0] += 0.16104959600493599;
                }
              }
            } else {
              if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.006413868905411052;
              } else {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.433569431304932529) ) ) {
                    result[0] += -0.03565539594912844;
                  } else {
                    if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                      result[0] += -0.024601005722870075;
                    } else {
                      result[0] += 0.0035052088653153603;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.174569487571716753) ) ) {
                      result[0] += -0.0606338684080579;
                    } else {
                      result[0] += 0.021642192324611027;
                    }
                  } else {
                    result[0] += 0.030557769761500504;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.071567356586456743) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
                  result[0] += 0.00842365834313334;
                } else {
                  result[0] += -0.061356256803687365;
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.28299736976623624) ) ) {
                  result[0] += -0.06847144475335384;
                } else {
                  result[0] += 0.0634113003836204;
                }
              }
            } else {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.029068946838379794) ) ) {
                  if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += -0.0006214997443799745;
                  } else {
                    result[0] += -0.03288365955865074;
                  }
                } else {
                  if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)7.338562726974488193) ) ) {
                    result[0] += -0.0019564481924345552;
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.921060562133789951) ) ) {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.216319084167481357) ) ) {
                        result[0] += 0.0042497281375222925;
                      } else {
                        result[0] += -0.020180996343884733;
                      }
                    } else {
                      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                        result[0] += 0.0003710826589211495;
                      } else {
                        if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.617236852645874912) ) ) {
                            result[0] += 0.004290191895487755;
                          } else {
                            result[0] += 0.050165474054047476;
                          }
                        } else {
                          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
                              if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
                                result[0] += 0.013621699878778454;
                              } else {
                                result[0] += 0.07160524959923562;
                              }
                            } else {
                              result[0] += 0.10356558800777294;
                            }
                          } else {
                            result[0] += 0.09192041217554024;
                          }
                        }
                      }
                    }
                  }
                }
              } else {
                result[0] += -0.025438942058519678;
              }
            }
          }
        }
      }
    } else {
      result[0] += -0.012075613229821694;
    }
  }
  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.617236852645874912) ) ) {
        result[0] += -0.001401398117171234;
      } else {
        result[0] += -0.014626253550399771;
      }
    } else {
      if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)2.393745899200439897) ) ) {
        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.182065486907959873) ) ) {
            result[0] += -0.00034219566385766936;
          } else {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += -0.018670632961735278;
              } else {
                result[0] += 0.012310596024202758;
              }
            } else {
              result[0] += 0.02188517499870918;
            }
          }
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
            result[0] += -0.0010080614795546239;
          } else {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)4.166635274887085849) ) ) {
                result[0] += -0.016420636190502887;
              } else {
                if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.835998296737671787) ) ) {
                    if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.531673669815064365) ) ) {
                        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.770631790161133257) ) ) {
                            if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                              result[0] += 0.04375150702527269;
                            } else {
                              result[0] += 0.13251680836541951;
                            }
                          } else {
                            result[0] += 0.0021905612977064282;
                          }
                        } else {
                          result[0] += -0.006934663811479245;
                        }
                      } else {
                        result[0] += 0.04481223839223207;
                      }
                    } else {
                      result[0] += -0.022097529002940347;
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.643222332000734198) ) ) {
                      result[0] += 0.004300767356540469;
                    } else {
                      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                        result[0] += -0.01494377984433411;
                      } else {
                        result[0] += -0.06103159578868216;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)12.00000000000000178) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
                      result[0] += -0.027734522822848696;
                    } else {
                      if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.025192260742188388) ) ) {
                            result[0] += 0.03517335591388572;
                          } else {
                            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.576439857482911933) ) ) {
                              result[0] += 0.00533109234403646;
                            } else {
                              result[0] += -0.04669012507924292;
                            }
                          }
                        } else {
                          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.342454433441162998) ) ) {
                            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.434600353240968573) ) ) {
                                result[0] += 0.10407830346382048;
                              } else {
                                result[0] += -0.0021648541746475797;
                              }
                            } else {
                              result[0] += -0.014898709607298808;
                            }
                          } else {
                            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.209340095520020419) ) ) {
                              result[0] += -0.024160212477630113;
                            } else {
                              if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.464211463928224433) ) ) {
                                  result[0] += 0.008009272684721434;
                                } else {
                                  result[0] += 0.10449237738699335;
                                }
                              } else {
                                result[0] += 0.030313230620610362;
                              }
                            }
                          }
                        }
                      } else {
                        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.81821727752685725) ) ) {
                          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                              result[0] += 0.04946076984135977;
                            } else {
                              result[0] += -0.01143652603865867;
                            }
                          } else {
                            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.158952236175537998) ) ) {
                              if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                                result[0] += 0.005883171331060241;
                              } else {
                                result[0] += -0.026901670710566677;
                              }
                            } else {
                              result[0] += 0.002602517924508301;
                            }
                          }
                        } else {
                          if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
                            if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
                              result[0] += 0.09537956360957973;
                            } else {
                              result[0] += 0.005451059036221203;
                            }
                          } else {
                            result[0] += 0.03792857749728656;
                          }
                        }
                      }
                    }
                  } else {
                    result[0] += 0.020480312887461938;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.141444921493531162) ) ) {
                  if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                    if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.481121778488159624) ) ) {
                      result[0] += -0.006185582600729939;
                    } else {
                      result[0] += -0.03537680747319587;
                    }
                  } else {
                    result[0] += 0.0026903358613589202;
                  }
                } else {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.941167116165162021) ) ) {
                        result[0] += 0.013442651396948745;
                      } else {
                        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.142630577087403232) ) ) {
                          result[0] += 0.0543514961164622;
                        } else {
                          result[0] += -0.04997438538072252;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += 0.04634032683427243;
                      } else {
                        result[0] += 0.0033545451559024325;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.674522399902344638) ) ) {
                      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                        if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.071567356586456743) ) ) {
                          result[0] += -0.03162633223931699;
                        } else {
                          result[0] += 0.02525209390722128;
                        }
                      } else {
                        if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.242453336715698464) ) ) {
                          result[0] += 0.013340528005088299;
                        } else {
                          result[0] += 0.10741957215513759;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                        result[0] += 0.010632859732775134;
                      } else {
                        if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.071567356586456743) ) ) {
                          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
                            if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                              result[0] += 0.04651676797312111;
                            } else {
                              result[0] += -0.01235194120615701;
                            }
                          } else {
                            result[0] += 0.04533611959965999;
                          }
                        } else {
                          result[0] += -0.02829266733851481;
                        }
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.020127415657043901) ) ) {
                    result[0] += 0.015254675700941966;
                  } else {
                    result[0] += -0.003803733208779768;
                  }
                } else {
                  if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += -0.00586432544802622;
                  } else {
                    if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.03662530833793863;
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.39772605895996271) ) ) {
                        result[0] += 0.01083068237459306;
                      } else {
                        result[0] += -0.03241477472076246;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
          result[0] += -0.021719284940000652;
        } else {
          result[0] += -0.15313600823101414;
        }
      }
    }
  } else {
    result[0] += 0.000944864813013456;
  }
  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)7.971558809280396396) ) ) {
    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.426736354827881748) ) ) {
        result[0] += 0.0009050260261764878;
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.30853915214538663) ) ) {
          result[0] += -0.020632861508357667;
        } else {
          result[0] += -0.0014999099396546948;
        }
      }
    } else {
      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.16594791412353693) ) ) {
        if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.24173307418823331) ) ) {
            result[0] += 0.007102967094925388;
          } else {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.12508964538574396) ) ) {
                  result[0] += -0.014409645132243862;
                } else {
                  result[0] += 0.02249580622001082;
                }
              } else {
                if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.737386107444763628) ) ) {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += 0.05367342310050264;
                    } else {
                      result[0] += -0.0567784289215264;
                    }
                  } else {
                    result[0] += 0.015771335590941505;
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.674522399902344638) ) ) {
                    result[0] += 0.03007848542677063;
                  } else {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
                      if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.384246587753296343) ) ) {
                        result[0] += 0.08908773442548101;
                      } else {
                        result[0] += -0.03334236548799519;
                      }
                    } else {
                      result[0] += -0.04713286173460385;
                    }
                  }
                }
              }
            } else {
              result[0] += -0.026157773319893177;
            }
          }
        } else {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.055496215820313388) ) ) {
            result[0] += 0.0012023698933808851;
          } else {
            result[0] += -0.0010343602644003404;
          }
        }
      } else {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
            if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
              if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.594915628433228427) ) ) {
                  result[0] += -0.009313918974622656;
                } else {
                  if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.993164777755738193) ) ) {
                    result[0] += 0.01321385001322459;
                  } else {
                    result[0] += -0.0018796644416854353;
                  }
                }
              } else {
                result[0] += -0.021748099195815447;
              }
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.321723937988282138) ) ) {
                if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                  result[0] += -0.0048333144707217645;
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.255632162094117099) ) ) {
                    result[0] += -0.031682697174924515;
                  } else {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += 0.032634920627227636;
                    } else {
                      result[0] += -0.015598691917038724;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.041798145833367904;
                } else {
                  result[0] += 0.06383294833904267;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.531673669815064365) ) ) {
                      if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.553712725639343706) ) ) {
                          result[0] += 0.0492830297685345;
                        } else {
                          result[0] += -0.09697533565139854;
                        }
                      } else {
                        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.431901693344116655) ) ) {
                          result[0] += -0.06702636053499399;
                        } else {
                          result[0] += 0.07947514414923224;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                        result[0] += 0.007390052806140558;
                      } else {
                        result[0] += -0.09815273645521579;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[10].missing != -1) || (data[10].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += 0.0067579304285634;
                    } else {
                      result[0] += -0.05109945273049491;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)3.000000000000000444) ) ) {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += -0.08755382022947311;
                    } else {
                      result[0] += -0.007827874219248358;
                    }
                  } else {
                    if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                        result[0] += 0.011350526313692622;
                      } else {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.777633190155030185) ) ) {
                          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.342454433441162998) ) ) {
                            result[0] += 0.017030346937851517;
                          } else {
                            result[0] += 0.07688129292649276;
                          }
                        } else {
                          result[0] += -0.019131274180197992;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.778982400894165927) ) ) {
                          result[0] += 0.007749047409293656;
                        } else {
                          result[0] += -0.06002046933211235;
                        }
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.18134641647339045) ) ) {
                          result[0] += -0.07769857334102573;
                        } else {
                          result[0] += 0.007441608372109955;
                        }
                      }
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.53326439857482999) ) ) {
                  result[0] += -0.00109688353081796;
                } else {
                  result[0] += 0.07674463015066384;
                }
              }
            } else {
              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += 0.0009292666281950283;
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += 0.02130266879572816;
                  } else {
                    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                      result[0] += 0.025102123686540214;
                    } else {
                      result[0] += -0.029672442759850655;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                    if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.357691764831543413) ) ) {
                      result[0] += 0.02325459951574681;
                    } else {
                      result[0] += -0.03122346339957683;
                    }
                  } else {
                    result[0] += -0.051515199243161704;
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.384830474853516513) ) ) {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += -0.02441956647254929;
              } else {
                result[0] += -0.0001448383186468207;
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.450390577316285068) ) ) {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)6.218359947204590732) ) ) {
                  result[0] += 1.745285133927671;
                } else {
                  result[0] += 0.007738444231653961;
                }
              } else {
                result[0] += 0.03441645315048212;
              }
            }
          } else {
            if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.007310603743336468;
            } else {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                result[0] += -0.02094423760939715;
              } else {
                result[0] += -0.046591591436520405;
              }
            }
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)12.00000000000000178) ) ) {
      result[0] += 0.0052725642178943315;
    } else {
      result[0] += -0.166507749527798;
    }
  }
  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)7.971558809280396396) ) ) {
    if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.029068946838379794) ) ) {
        result[0] += 0.0010700440799398814;
      } else {
        if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.992907285690308505) ) ) {
            result[0] += 7.540378742152742e-05;
          } else {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
              result[0] += -0.08061177442221787;
            } else {
              if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.002201207024911768;
              } else {
                if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.007936503457548585;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.921924352645874468) ) ) {
                    result[0] += 0.07515225948262705;
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.82428741455078303) ) ) {
                      result[0] += -0.039519935684717594;
                    } else {
                      result[0] += 0.037369655517283926;
                    }
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.766185760498047763) ) ) {
              if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
                result[0] += -0.01448733780884248;
              } else {
                result[0] += -0.04069478511934665;
              }
            } else {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                    if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.497866153717041238) ) ) {
                      result[0] += -0.028731374844433777;
                    } else {
                      result[0] += -0.08514372063562624;
                    }
                  } else {
                    if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
                      if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.868834793567657693) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.596743106842042792) ) ) {
                          result[0] += 0.11273406381799395;
                        } else {
                          result[0] += 0.004606981523168452;
                        }
                      } else {
                        result[0] += 0.16187977804382006;
                      }
                    } else {
                      if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
                        if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                          if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                              result[0] += -0.04907565244066788;
                            } else {
                              result[0] += -6.429001404503839e-05;
                            }
                          } else {
                            if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.880305767059327948) ) ) {
                                if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                                  result[0] += -0.05089131936557632;
                                } else {
                                  result[0] += 0.008536538491925075;
                                }
                              } else {
                                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.467917680740357333) ) ) {
                                  result[0] += -0.019433308532368104;
                                } else {
                                  if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                                    result[0] += 0.05783513403097789;
                                  } else {
                                    result[0] += -0.009120945869309869;
                                  }
                                }
                              }
                            } else {
                              result[0] += -0.025113904469085537;
                            }
                          }
                        } else {
                          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                            result[0] += 0.1329698398677691;
                          } else {
                            result[0] += -0.05138770891819681;
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                          result[0] += 0.11507775710302195;
                        } else {
                          if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                            result[0] += -0.08889116163875785;
                          } else {
                            if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.129780292510988104) ) ) {
                                result[0] += -0.1115239348120914;
                              } else {
                                result[0] += 0.05912481616230841;
                              }
                            } else {
                              result[0] += -0.07611093014858461;
                            }
                          }
                        }
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.602003335952759233) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.81940793991089045) ) ) {
                      result[0] += 0.015327577461239715;
                    } else {
                      result[0] += -0.0729325676930239;
                    }
                  } else {
                    result[0] += -0.011611114695556498;
                  }
                }
              } else {
                result[0] += 0.00039157235101152364;
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.056097030639650214) ) ) {
              result[0] += 0.0033582289661210443;
            } else {
              if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += -0.006072576944129867;
                } else {
                  result[0] += 0.03185345678661643;
                }
              } else {
                if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.023493805460725117;
                } else {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                    if ( UNLIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.010363595738398781;
                    } else {
                      if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                        result[0] += -0.0565896683256385;
                      } else {
                        result[0] += -0.016228467068688704;
                      }
                    }
                  } else {
                    result[0] += 0.003306481251683679;
                  }
                }
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.198252916336060458) ) ) {
        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
          result[0] += 0.0026274997702685566;
        } else {
          if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.242453336715698464) ) ) {
            result[0] += -0.00246853077114641;
          } else {
            result[0] += 0.010216343403796686;
          }
        }
      } else {
        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
            result[0] += -0.03442409146875946;
          } else {
            result[0] += -0.0023548349788423467;
          }
        } else {
          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
            if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.992907285690308505) ) ) {
                result[0] += -0.006247357307169279;
              } else {
                result[0] += -0.06032477291664569;
              }
            } else {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += 0.01714041841584914;
              } else {
                result[0] += 0.05715941486533417;
              }
            }
          } else {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)12.00000000000000178) ) ) {
              if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.422742605209351474) ) ) {
                result[0] += -0.0011815992266169147;
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.342454433441162998) ) ) {
                  result[0] += -0.005295016687439993;
                } else {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.004577280442922199;
                    } else {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.379217386245728427) ) ) {
                        result[0] += -0.002690938661865467;
                      } else {
                        result[0] += 0.019481361509895406;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.601370334625245029) ) ) {
                        if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                          result[0] += 0.000647795330016483;
                        } else {
                          result[0] += -0.015896091599011102;
                        }
                      } else {
                        result[0] += 0.012747887888760807;
                      }
                    } else {
                      result[0] += 0.005840109415743808;
                    }
                  }
                }
              }
            } else {
              result[0] += 0.004948530069583203;
            }
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)12.00000000000000178) ) ) {
      result[0] += 0.004081454467624697;
    } else {
      result[0] += -0.16255484460714037;
    }
  }
  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
    result[0] += 8.14393052004612e-05;
  } else {
    if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.740319490432739702) ) ) {
      if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
        if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.700598716735840066) ) ) {
          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
            if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.094205617904663974) ) ) {
                  result[0] += -0.06357543709001416;
                } else {
                  if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.350240230560303178) ) ) {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.136462926864624912) ) ) {
                      if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                        result[0] += 0.030263908550661852;
                      } else {
                        result[0] += 0.1042692615924633;
                      }
                    } else {
                      result[0] += -0.01811025911091014;
                    }
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.909254074096680576) ) ) {
                      result[0] += 0.009320504580524339;
                    } else {
                      result[0] += 0.21038775079521954;
                    }
                  }
                }
              } else {
                result[0] += 0.009689545666950862;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.867504835128785068) ) ) {
                result[0] += 0.003185272984376198;
              } else {
                result[0] += -0.0405734122947328;
              }
            }
          } else {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1.497866153717041238) ) ) {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.068990230560303623) ) ) {
                      result[0] += 0.07675550643206933;
                    } else {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
                        result[0] += 0.10169176244447298;
                      } else {
                        result[0] += -0.06712258077844573;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.071567356586456743) ) ) {
                      result[0] += -0.15534948109291916;
                    } else {
                      result[0] += 0.0024657127331109887;
                    }
                  }
                } else {
                  result[0] += -0.05181605172066161;
                }
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.637949228286744052) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.094205617904663974) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.835998296737671787) ) ) {
                      result[0] += -0.06486380686463467;
                    } else {
                      result[0] += 0.07233048005433675;
                    }
                  } else {
                    result[0] += -0.05237240943418611;
                  }
                } else {
                  if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
                      result[0] += 0.021476975125622987;
                    } else {
                      result[0] += -0.03767021863696444;
                    }
                  } else {
                    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += 0.035465220860157146;
                    } else {
                      result[0] += -0.05044411548014466;
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.700598716735840066) ) ) {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.467917680740357333) ) ) {
                      result[0] += 0.013356589715299384;
                    } else {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.53326439857482999) ) ) {
                        result[0] += 0.19663973085317915;
                      } else {
                        result[0] += 0.02707610667357234;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.102759599685669833) ) ) {
                      result[0] += -0.0227532182934625;
                    } else {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.745876312255860263) ) ) {
                        result[0] += 0.08380778001561244;
                      } else {
                        result[0] += -0.00499924234655075;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.44140100479126021) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.802696108818054643) ) ) {
                      result[0] += 0.12538092868938863;
                    } else {
                      result[0] += 0.03671566565451374;
                    }
                  } else {
                    result[0] += 0.003131549102465009;
                  }
                }
              } else {
                if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                    if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
                      result[0] += -0.01649340488887342;
                    } else {
                      result[0] += 0.16758173455691913;
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.921924352645874468) ) ) {
                      result[0] += 0.062343297796760844;
                    } else {
                      result[0] += 0.0017373903659320877;
                    }
                  }
                } else {
                  result[0] += -0.057735714501197015;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.715336322784424716) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.737386107444763628) ) ) {
              result[0] += -0.03664927210887354;
            } else {
              result[0] += 0.04246528353130411;
            }
          } else {
            result[0] += -0.056357699949870214;
          }
        }
      } else {
        result[0] += -0.05558973520469691;
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.493027687072754794) ) ) {
        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.249904870986938921) ) ) {
          if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.861792564392090288) ) ) {
            result[0] += -0.0475219537126773;
          } else {
            if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
              if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.921060562133789951) ) ) {
                  if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.248013019561768466) ) ) {
                      result[0] += 0.2393964675024486;
                    } else {
                      result[0] += 0.025618541190248013;
                    }
                  } else {
                    result[0] += 0.11830112520275445;
                  }
                } else {
                  result[0] += -0.027337358013139114;
                }
              } else {
                if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.094205617904663974) ) ) {
                      result[0] += 0.09336558696994997;
                    } else {
                      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.219419956207276279) ) ) {
                        result[0] += -0.005177161882522344;
                      } else {
                        result[0] += 0.061167985835503584;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.248013019561768466) ) ) {
                      result[0] += 0.19185673663181446;
                    } else {
                      result[0] += -0.0328090429475726;
                    }
                  }
                } else {
                  result[0] += -0.043497445540850076;
                }
              }
            } else {
              result[0] += -0.06123008150839503;
            }
          }
        } else {
          result[0] += -0.04541558300201401;
        }
      } else {
        if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.962127923965454546) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.750972747802735263) ) ) {
              result[0] += 0.006266587403168777;
            } else {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.450390577316285068) ) ) {
                result[0] += -0.05124491245345624;
              } else {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                  result[0] += -0.06175560840209097;
                } else {
                  if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.022051495736373602;
                  } else {
                    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += 0.03062975153174943;
                    } else {
                      result[0] += -0.061359406155598685;
                    }
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += -0.07794780603956702;
            } else {
              result[0] += 0.1015618634471843;
            }
          }
        } else {
          result[0] += -0.06369025396815171;
        }
      }
    }
  }
  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.219399690628052646) ) ) {
      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.60838890075683771) ) ) {
        result[0] += 0.0006609857831823228;
      } else {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
          result[0] += 0.00741340472501156;
        } else {
          result[0] += -0.014269297093598022;
        }
      }
    } else {
      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.54220247268676935) ) ) {
        result[0] += -0.00033725824578352505;
      } else {
        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
          if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
            result[0] += 0.009714018514800506;
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.450390577316285068) ) ) {
              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.700598716735840066) ) ) {
                result[0] += -0.015079036510258173;
              } else {
                result[0] += -0.06597914860914815;
              }
            } else {
              result[0] += -0.06442651013919276;
            }
          }
        } else {
          result[0] += -0.0035524997218048314;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.189540147781372958) ) ) {
      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.417592287063599077) ) ) {
        if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)10.11102914810180842) ) ) {
          if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += 0.0198943769701593;
          } else {
            if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)257681260544.0000305) ) ) {
              result[0] += -0.023215628673174334;
            } else {
              result[0] += 0.008383695927363651;
            }
          }
        } else {
          result[0] += -0.03734907715130035;
        }
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)1.497866153717041238) ) ) {
          result[0] += 0.03363267613480191;
        } else {
          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
            result[0] += -0.050775710784252394;
          } else {
            result[0] += -0.02694229732612547;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.96495962142944514) ) ) {
        if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.868834793567657693) ) ) {
          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.766185760498047763) ) ) {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.334978580474854404) ) ) {
                  if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.022730382370902885;
                  } else {
                    result[0] += -0.11411021001220592;
                  }
                } else {
                  result[0] += 0.012990349768087743;
                }
              } else {
                result[0] += 0.02606613085233264;
              }
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.507949829101563388) ) ) {
                if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += 0.026453426682263487;
                  } else {
                    result[0] += -0.01860148917943497;
                  }
                } else {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += 0.03166826752610068;
                  } else {
                    result[0] += 0.10362919197364823;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += 0.04721411706581917;
                } else {
                  result[0] += 0.1570853342722408;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
              if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.05259225310243344;
              } else {
                result[0] += 0.012666866471632158;
              }
            } else {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)9.520321369171144354) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.795762062072754794) ) ) {
                    if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.637949228286744052) ) ) {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
                        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.266057968139650214) ) ) {
                              result[0] += 0.13763855901263342;
                            } else {
                              result[0] += -0.013179288699174533;
                            }
                          } else {
                            result[0] += 0.007580227824635154;
                          }
                        } else {
                          result[0] += -0.015414256151301564;
                        }
                      } else {
                        if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += -0.00027355358916563186;
                        } else {
                          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            result[0] += -0.03458350241388449;
                          } else {
                            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.40472650527954279) ) ) {
                              result[0] += -0.010079738089604466;
                            } else {
                              result[0] += 0.02252453756704093;
                            }
                          }
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
                        result[0] += -0.020124297143248836;
                      } else {
                        result[0] += -0.06717068344966358;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.868834793567657693) ) ) {
                      if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)3.449861526489258257) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.511434078216553178) ) ) {
                          result[0] += -0.0005960181630413042;
                        } else {
                          result[0] += -0.026794037240462955;
                        }
                      } else {
                        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.827801465988160068) ) ) {
                          result[0] += 0.034054490904370405;
                        } else {
                          if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.146417617797853339) ) ) {
                            result[0] += 0.0031305910622868426;
                          } else {
                            result[0] += 0.0262123972195383;
                          }
                        }
                      }
                    } else {
                      result[0] += 0.08422079409710016;
                    }
                  }
                } else {
                  result[0] += 0.022921733114196787;
                }
              } else {
                if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.35311269760132014) ) ) {
                      if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += -0.02005739643864976;
                      } else {
                        result[0] += -0.06801421931654815;
                      }
                    } else {
                      result[0] += -0.003119589090711634;
                    }
                  } else {
                    if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.024506205460804332;
                    } else {
                      result[0] += 0.006740648120793106;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.740319490432739702) ) ) {
                    result[0] += 0.023140641776106433;
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.847591876983644354) ) ) {
                      result[0] += -0.020280763235389528;
                    } else {
                      if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)257681260544.0000305) ) ) {
                        result[0] += 0.04465014430339539;
                      } else {
                        result[0] += -0.005549372706640576;
                      }
                    }
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.347943067550660068) ) ) {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
              result[0] += 0.019564862897522614;
            } else {
              result[0] += -0.02542002065286275;
            }
          } else {
            result[0] += -0.032096806665515405;
          }
        }
      } else {
        if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
          if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += 0.01877819728316874;
            } else {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.993164777755738193) ) ) {
                result[0] += -0.03178839914404288;
              } else {
                result[0] += 0.026689633134861324;
              }
            }
          } else {
            if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += 0.03892496829399197;
            } else {
              result[0] += -0.027981901242432456;
            }
          }
        } else {
          result[0] += -0.050217918765617035;
        }
      }
    }
  }
  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
    result[0] += -0.00022763048639246975;
  } else {
    if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.189540147781372958) ) ) {
        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.417592287063599077) ) ) {
          if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)10.03981685638427912) ) ) {
            if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.02107086668700704;
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.241523027420044833) ) ) {
                result[0] += 0.003961611429328179;
              } else {
                result[0] += -0.04515308539062063;
              }
            }
          } else {
            result[0] += -0.029417010502546404;
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.154959201812744585) ) ) {
            result[0] += 0.011020024690037896;
          } else {
            result[0] += -0.029278314697299752;
          }
        }
      } else {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.29926824569702326) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.43450713157653853) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.795762062072754794) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
                result[0] += 0.004153018130670747;
              } else {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += 0.0033997476397551966;
                  } else {
                    if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)3072.000000000000455) ) ) {
                      result[0] += 0.01657303677512327;
                    } else {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.637949228286744052) ) ) {
                        if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.43749904632568537) ) ) {
                            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                              result[0] += -0.12258155480949628;
                            } else {
                              result[0] += -0.03827268699353267;
                            }
                          } else {
                            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                              result[0] += 0.007378262869434978;
                            } else {
                              result[0] += -0.023172396318781534;
                            }
                          }
                        } else {
                          result[0] += -0.008563668223726787;
                        }
                      } else {
                        result[0] += -0.0460801219559235;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += 0.039197234851827654;
                  } else {
                    result[0] += 0.00036110055688373384;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.940167903900147373) ) ) {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += -0.07920702856666345;
                  } else {
                    result[0] += 0.026559595712597167;
                  }
                } else {
                  if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += -0.013627474987010516;
                  } else {
                    result[0] += 0.042033205420438105;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)3.449861526489258257) ) ) {
                  result[0] += -0.011791442531534949;
                } else {
                  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += 0.04139918439621437;
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.636499762535095659) ) ) {
                      result[0] += 0.03283104962624941;
                    } else {
                      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                        if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.384246587753296343) ) ) {
                          result[0] += 0.001031247613720583;
                        } else {
                          if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                            if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                              if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.057953596115113193) ) ) {
                                if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.435900688171387607) ) ) {
                                  result[0] += 0.015563777429770505;
                                } else {
                                  result[0] += -0.02992310175943246;
                                }
                              } else {
                                result[0] += 0.01700459031226832;
                              }
                            } else {
                              if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                                result[0] += 0.02575073889250449;
                              } else {
                                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.012675821781158891) ) ) {
                                  result[0] += 0.034604422861609536;
                                } else {
                                  result[0] += -0.03949854491207579;
                                }
                              }
                            }
                          } else {
                            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.700598716735840066) ) ) {
                              if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.855921268463135654) ) ) {
                                result[0] += -0.013975589642471704;
                              } else {
                                result[0] += 0.03203504608876023;
                              }
                            } else {
                              if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                                result[0] += 0.018043903386149626;
                              } else {
                                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.329314231872559482) ) ) {
                                    result[0] += 0.07669843848331115;
                                  } else {
                                    result[0] += -0.04237701302871186;
                                  }
                                } else {
                                  result[0] += 0.05316927343625325;
                                }
                              }
                            }
                          }
                        }
                      } else {
                        if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.597218394279480425) ) ) {
                          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.750972747802735263) ) ) {
                                result[0] += 0.043049019868594586;
                              } else {
                                if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                                  result[0] += -0.03834982991010866;
                                } else {
                                  result[0] += 0.01687236733714008;
                                }
                              }
                            } else {
                              result[0] += -0.04955216834826076;
                            }
                          } else {
                            if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)3.000000000000000444) ) ) {
                              result[0] += 0.02883340574782713;
                            } else {
                              result[0] += 0.004180446607650875;
                            }
                          }
                        } else {
                          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                            if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                              result[0] += -0.03201920232601289;
                            } else {
                              result[0] += 0.018825495572450335;
                            }
                          } else {
                            result[0] += -0.028631104861216762;
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
              result[0] += 0.004130021857477176;
            } else {
              result[0] += 0.08715558047468108;
            }
          }
        } else {
          if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                result[0] += 0.024400225927604876;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.288152217864991123) ) ) {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
                      result[0] += -0.020253173953023368;
                    } else {
                      result[0] += -0.08906824630400542;
                    }
                  } else {
                    result[0] += 0.019732246590552685;
                  }
                } else {
                  result[0] += 0.015096496502550974;
                }
              }
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.993164777755738193) ) ) {
                if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                  result[0] += -0.03665591702520731;
                } else {
                  result[0] += 0.010112370432834325;
                }
              } else {
                result[0] += 0.01576563508795006;
              }
            }
          } else {
            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
              result[0] += 0.021634036629016355;
            } else {
              result[0] += 0.05403892102618074;
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.417592287063599077) ) ) {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
          if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
            result[0] += -0.010648848829159889;
          } else {
            result[0] += 0.02499043078501979;
          }
        } else {
          result[0] += -0.032758047700376505;
        }
      } else {
        result[0] += -0.04306036074915873;
      }
    }
  }
  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
    result[0] += -0.00022396409558110482;
  } else {
    if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.288152217864991123) ) ) {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.189540147781372958) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.434600353240968573) ) ) {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.142630577087403232) ) ) {
                  result[0] += 0.06217752253516571;
                } else {
                  result[0] += -0.009608383600744973;
                }
              } else {
                result[0] += 0.006637840031639585;
              }
            } else {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.241523027420044833) ) ) {
                    result[0] += -0.003643027714575817;
                  } else {
                    result[0] += -0.05933090024183903;
                  }
                } else {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.08127461933078818;
                  } else {
                    result[0] += -0.03085099484581277;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                  result[0] += -0.05360224258632282;
                } else {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.417592287063599077) ) ) {
                    result[0] += 0.002505608854226052;
                  } else {
                    result[0] += -0.01767565185404881;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.60838890075683771) ) ) {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += 0.008635669011148059;
              } else {
                result[0] += -0.00573590776351001;
              }
            } else {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
                if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.700598716735840066) ) ) {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += 0.02222703588045804;
                    } else {
                      result[0] += 0.056860645718837616;
                    }
                  } else {
                    result[0] += -0.011790660801804171;
                  }
                } else {
                  result[0] += 0.08973877709639395;
                }
              } else {
                result[0] += -0.017070135216001866;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.82428741455078303) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
              result[0] += 0.01981211972208682;
            } else {
              result[0] += -0.014531017281462666;
            }
          } else {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
              if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += -0.17590547783563387;
              } else {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                  result[0] += -0.04346299912247776;
                } else {
                  result[0] += -0.08398871647773082;
                }
              }
            } else {
              result[0] += 0.013592022195338675;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.26837396621704279) ) ) {
          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.97887301445007413) ) ) {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += 0.005693986425444771;
              } else {
                result[0] += -0.07807568545817949;
              }
            } else {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += 0.02976786001604556;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.625595092773438388) ) ) {
                    result[0] += -0.09690444812356344;
                  } else {
                    result[0] += 0.0029489852728094724;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.53326439857482999) ) ) {
                  result[0] += -0.050013575334326245;
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.102759599685669833) ) ) {
                    result[0] += 0.06028794202706318;
                  } else {
                    result[0] += 0.13397138963683275;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.43450713157653853) ) ) {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.24173307418823331) ) ) {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                    result[0] += -0.08093546852762537;
                  } else {
                    result[0] += 0.04084928111311795;
                  }
                } else {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                      result[0] += -0.0858051455554867;
                    } else {
                      result[0] += 0.038208045271438756;
                    }
                  } else {
                    result[0] += 0.06063272814140574;
                  }
                }
              } else {
                if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)9.520321369171144354) ) ) {
                    result[0] += 0.0002664655028970515;
                  } else {
                    result[0] += 0.031224491114192413;
                  }
                } else {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                      if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                        if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.625595092773438388) ) ) {
                            result[0] += -0.042666904424000084;
                          } else {
                            if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
                              result[0] += 0.005265758105183687;
                            } else {
                              result[0] += 0.1043928892976839;
                            }
                          }
                        } else {
                          if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                            result[0] += -0.09082421996889216;
                          } else {
                            result[0] += -0.014456817887232704;
                          }
                        }
                      } else {
                        result[0] += -0.08299450564029198;
                      }
                    } else {
                      result[0] += -0.09622095542598263;
                    }
                  } else {
                    if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                        result[0] += -0.0032885140108686083;
                      } else {
                        result[0] += -0.054724012127387846;
                      }
                    } else {
                      result[0] += 0.01073420593297534;
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                result[0] += 0.0036094169305726817;
              } else {
                result[0] += 0.10333481786445738;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += 0.011164236313142149;
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.780892848968506748) ) ) {
                if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                  result[0] += 0.030129230491945536;
                } else {
                  result[0] += -0.023787593517082622;
                }
              } else {
                result[0] += 0.03270787500507124;
              }
            }
          } else {
            if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
              if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                    if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.017104411154262044;
                    } else {
                      result[0] += 0.04516343660253822;
                    }
                  } else {
                    result[0] += 0.1013331569869419;
                  }
                } else {
                  result[0] += 0.09863603632396842;
                }
              } else {
                result[0] += 0.02384607824125784;
              }
            } else {
              result[0] += -0.03353169923639124;
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.749434947967529741) ) ) {
        result[0] += -0.009896252735239364;
      } else {
        result[0] += -0.04117368809793585;
      }
    }
  }
  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
    result[0] += -0.001283385059493291;
  } else {
    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
      if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)3.000000000000000444) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.597137451171875888) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.00396752357482999) ) ) {
            if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.0014781433791921377;
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.381086945533752885) ) ) {
                result[0] += -0.043707787585797374;
              } else {
                result[0] += 0.05361131791150117;
              }
            }
          } else {
            result[0] += -0.020736758275217156;
          }
        } else {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.531673669815064365) ) ) {
            result[0] += -0.0011493349310700502;
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.537586927413940874) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.397998809814454013) ) ) {
                result[0] += -0.0009147052251119235;
              } else {
                result[0] += -0.029791249860071846;
              }
            } else {
              result[0] += -0.10297548124233212;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)8.285748958587648261) ) ) {
          result[0] += -0.00018441518262831284;
        } else {
          result[0] += -0.030111340124412342;
        }
      }
    } else {
      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)8.285748958587648261) ) ) {
        if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)3.449861526489258257) ) ) {
          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.837713479995728427) ) ) {
              if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.00580711499603621;
                  } else {
                    result[0] += 0.1262955237618147;
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)1.497866153717041238) ) ) {
                    result[0] += -0.10805154066710457;
                  } else {
                    result[0] += 0.017730207907420175;
                  }
                }
              } else {
                result[0] += 0.04724665382856477;
              }
            } else {
              result[0] += -0.01821440579931031;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.921924352645874468) ) ) {
              result[0] += -0.00018198442544930268;
            } else {
              result[0] += -0.018852281131199286;
            }
          }
        } else {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
              result[0] += -0.04032154605250601;
            } else {
              result[0] += 0.024305035043038346;
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.921060562133789951) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.9055976867675799) ) ) {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.637949228286744052) ) ) {
                    result[0] += -0.0008713482092900972;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.567862033843995029) ) ) {
                      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                        result[0] += 0.02134348373429188;
                      } else {
                        if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
                          result[0] += -0.04376538878737897;
                        } else {
                          result[0] += 0.07394188376036105;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                        result[0] += 0.0028403896588744043;
                      } else {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.637949228286744052) ) ) {
                          result[0] += -0.0128301755545805;
                        } else {
                          result[0] += -0.04383766617032858;
                        }
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.09806728363037287) ) ) {
                    if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += 0.009277271042701207;
                    } else {
                      result[0] += -0.019082485312180434;
                    }
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.993164777755738193) ) ) {
                      result[0] += -0.022025500858964833;
                    } else {
                      result[0] += 0.022003227120951385;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
                  if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)6.553220748901368076) ) ) {
                    result[0] += -0.029726990018551315;
                  } else {
                    result[0] += 0.03525758848462377;
                  }
                } else {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += 0.01761015742613145;
                  } else {
                    result[0] += -0.015564073122408463;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.855006217956543857) ) ) {
                  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.007194855189064562;
                  } else {
                    result[0] += -0.0006415561498651014;
                  }
                } else {
                  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                      if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)6.000000000000000888) ) ) {
                        result[0] += -0.010635076879475864;
                      } else {
                        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
                          result[0] += -0.004842125106154862;
                        } else {
                          if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
                            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2252.000000000000455) ) ) {
                              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.511434078216553178) ) ) {
                                result[0] += 0.027510270972777064;
                              } else {
                                result[0] += -0.09810072800203373;
                              }
                            } else {
                              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.589234352111818183) ) ) {
                                result[0] += 0.052749273840308766;
                              } else {
                                result[0] += 0.007826776254212027;
                              }
                            }
                          } else {
                            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.909102678298951083) ) ) {
                              result[0] += -0.03835875755218585;
                            } else {
                              result[0] += 0.04182101675526326;
                            }
                          }
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += -0.043977841441954435;
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.017797946929933417) ) ) {
                          result[0] += -0.06072739341924939;
                        } else {
                          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                            if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                              result[0] += 0.023807444315902142;
                            } else {
                              result[0] += -0.005752186030535943;
                            }
                          } else {
                            result[0] += 0.03161782239938753;
                          }
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.046861171722413886) ) ) {
                      result[0] += 0.0002992887378486309;
                    } else {
                      if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)3.000000000000000444) ) ) {
                        result[0] += 0.004282948782084454;
                      } else {
                        result[0] += 0.0354063106059842;
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.53326439857482999) ) ) {
                    result[0] += -0.02094555423339017;
                  } else {
                    result[0] += 0.047338328911472584;
                  }
                } else {
                  if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.511434078216553178) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.851041555404663974) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.556798219680787021) ) ) {
                        result[0] += 0.023712265662272886;
                      } else {
                        if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.242453336715698464) ) ) {
                          result[0] += 0.0026390163094356977;
                        } else {
                          result[0] += 0.05128421385419092;
                        }
                      }
                    } else {
                      result[0] += -0.006954104567952601;
                    }
                  } else {
                    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                      result[0] += 0.0033014793940393874;
                    } else {
                      result[0] += -0.020651786592169024;
                    }
                  }
                }
              }
            }
          }
        }
      } else {
        result[0] += 0.052676918209847094;
      }
    }
  }
  if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.00000001800250948e-35) ) ) {
    result[0] += 0.002796907095665359;
  } else {
    if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
      if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
        if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)3.000000000000000444) ) ) {
          result[0] += -0.04101539600193686;
        } else {
          result[0] += -0.009882035317090011;
        }
      } else {
        if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)8816427008.000001907) ) ) {
          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
            result[0] += 0.029995289677135807;
          } else {
            result[0] += -0.14426222097802754;
          }
        } else {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.18088722229004084) ) ) {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.992907285690308505) ) ) {
              if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                result[0] += -0.0020387877850275636;
              } else {
                result[0] += 0.001203999928098573;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.511434078216553178) ) ) {
                result[0] += 0.027312307358508897;
              } else {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += -0.0071829347564096166;
                } else {
                  result[0] += -0.033991443816010596;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.993164777755738193) ) ) {
                result[0] += -0.008096082945561262;
              } else {
                result[0] += -0.03311172072625568;
              }
            } else {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.636499762535095659) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.18965101242065607) ) ) {
                    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.14446605154067485;
                    } else {
                      result[0] += -0.04270925364073098;
                    }
                  } else {
                    result[0] += -0.0670704418957562;
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.938867926597595659) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.47345590591430842) ) ) {
                      result[0] += -0.1474460184970465;
                    } else {
                      result[0] += -0.015888123265750717;
                    }
                  } else {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.605039834976196733) ) ) {
                      result[0] += 0.033412691062505266;
                    } else {
                      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                        result[0] += 0.001260049232891006;
                      } else {
                        result[0] += 0.04417841416485283;
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.433569431304932529) ) ) {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                      if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.173316955566407138) ) ) {
                        if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += 0.001249497501049017;
                        } else {
                          result[0] += -0.03237374533018524;
                        }
                      } else {
                        if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)2.012675821781158891) ) ) {
                          result[0] += 0.031058255136805508;
                        } else {
                          result[0] += -0.05103662355460212;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.737386107444763628) ) ) {
                        result[0] += 0.03882180660944138;
                      } else {
                        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.918272972106934482) ) ) {
                          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.69067406654357999) ) ) {
                            result[0] += -0.04922107198593511;
                          } else {
                            if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
                              result[0] += 0.03438938103009969;
                            } else {
                              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.14095449447632014) ) ) {
                                result[0] += -0.10212668261224343;
                              } else {
                                result[0] += 0.015487706556003844;
                              }
                            }
                          }
                        } else {
                          result[0] += -0.08693514450551883;
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.636499762535095659) ) ) {
                        result[0] += -0.023830027049815135;
                      } else {
                        result[0] += 0.01347208228099838;
                      }
                    } else {
                      result[0] += -0.0513683110267971;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += -0.029466619648930045;
                  } else {
                    result[0] += 0.03209208213896591;
                  }
                }
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
          result[0] += -9.167227416565483e-05;
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.97887301445007413) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.909254074096680576) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.029068946838379794) ) ) {
                result[0] += -0.012958448118742711;
              } else {
                result[0] += 0.002554376301455507;
              }
            } else {
              result[0] += 0.022412331576377842;
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.938867926597595659) ) ) {
              result[0] += 0.17692527168233574;
            } else {
              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.367881059646607333) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.198252916336060458) ) ) {
                    result[0] += 0.003502177780909095;
                  } else {
                    result[0] += -0.02958495999953292;
                  }
                } else {
                  result[0] += 0.023437703195719955;
                }
              } else {
                if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.602003335952759233) ) ) {
                  result[0] += -0.05289288975282663;
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.025192260742188388) ) ) {
                    result[0] += 0.07541207879131383;
                  } else {
                    result[0] += -0.022257682144664104;
                  }
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
          if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.497866153717041238) ) ) {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.302512168884278232) ) ) {
              if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5.424940347671509677) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)15.69420576095581232) ) ) {
                  result[0] += 0.0031737658207378334;
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)16.36023521423340199) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.431901693344116655) ) ) {
                      result[0] += -0.17215458337833525;
                    } else {
                      result[0] += 0.004260441180970864;
                    }
                  } else {
                    if ( UNLIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.12466522038883483;
                    } else {
                      result[0] += 0.04310940211453326;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.770631790161133257) ) ) {
                  result[0] += 0.10560151393428874;
                } else {
                  result[0] += 0.007079924099613586;
                }
              }
            } else {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.725620865821838823) ) ) {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += 0.07411067799702294;
                    } else {
                      result[0] += -0.027446583099475275;
                    }
                  } else {
                    result[0] += -0.06581659706966424;
                  }
                } else {
                  result[0] += 0.057989131776482405;
                }
              } else {
                if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.725620865821838823) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.87502956390381037) ) ) {
                      result[0] += 0.0005456521316219576;
                    } else {
                      result[0] += 0.1277278806348595;
                    }
                  } else {
                    result[0] += 0.1030262180268631;
                  }
                } else {
                  result[0] += -0.007277485982956417;
                }
              }
            }
          } else {
            result[0] += -0.03630937926124186;
          }
        } else {
          result[0] += 0.003174070303398894;
        }
      }
    }
  }
  if ( UNLIKELY(  (data[37].missing != -1) && (data[37].fvalue <= (double)-1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
      result[0] += 0.006129687468278986;
    } else {
      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
        result[0] += 0.0003329384049511701;
      } else {
        if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.03420138359069913) ) ) {
          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.102759599685669833) ) ) {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += -0.12769565690213833;
              } else {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.15100884437561124) ) ) {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.467917680740357333) ) ) {
                    result[0] += -0.07513931619252885;
                  } else {
                    result[0] += 0.015677704207289176;
                  }
                } else {
                  result[0] += -0.07201179005912703;
                }
              }
            } else {
              result[0] += 0.03256755193648708;
            }
          } else {
            if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.650573849678039995) ) ) {
              result[0] += 0.047289332616029606;
            } else {
              if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.962292194366455966) ) ) {
                result[0] += -0.08411091459295023;
              } else {
                result[0] += 0.0313058704204632;
              }
            }
          }
        } else {
          result[0] += -0.12656353204793216;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)8816427008.000001907) ) ) {
      if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
        result[0] += -0.06277843648098812;
      } else {
        result[0] += 0.0906185844141777;
      }
    } else {
      if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
          result[0] += -0.013434506798032299;
        } else {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
            result[0] += -0.000743370983640582;
          } else {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.53326439857482999) ) ) {
                result[0] += -0.04877898047367028;
              } else {
                result[0] += 0.10663794965420433;
              }
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.553712725639343706) ) ) {
                result[0] += 0.04746050004261361;
              } else {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                  result[0] += -0.06028116650834338;
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.434600353240968573) ) ) {
                    result[0] += -0.047432840909528894;
                  } else {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.344550132751465732) ) ) {
                      result[0] += -0.05675111592755603;
                    } else {
                      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += 0.09986953610633395;
                      } else {
                        result[0] += -0.020135238894089934;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.029068946838379794) ) ) {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.673553824424744096) ) ) {
                      result[0] += -0.006863350412600489;
                    } else {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.467917680740357333) ) ) {
                        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                          result[0] += 0.006354762843952898;
                        } else {
                          result[0] += 0.0365005674204094;
                        }
                      } else {
                        result[0] += 0.00399434549721552;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.006385558944918954;
                    } else {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.464467763900757724) ) ) {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.311204195022583896) ) ) {
                          result[0] += -0.01378112875340104;
                        } else {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.046861171722413886) ) ) {
                            result[0] += -0.0231950656132358;
                          } else {
                            result[0] += -0.1168310197430918;
                          }
                        }
                      } else {
                        result[0] += 0.06990531265945772;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.861792564392090288) ) ) {
                    result[0] += 0.01999873784445626;
                  } else {
                    result[0] += -0.002654402670065166;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.662244915962219682) ) ) {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += -0.022619623573229553;
                  } else {
                    result[0] += 0.011462186765372766;
                  }
                } else {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                    result[0] += 0.017399986721308602;
                  } else {
                    result[0] += -0.003005315365042656;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.388237953186036044) ) ) {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.877672910690308505) ) ) {
                    if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                      result[0] += 0.02038680348634349;
                    } else {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.851041555404663974) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.36986422538757413) ) ) {
                          result[0] += 0.10820141205414793;
                        } else {
                          result[0] += -0.015307171027276243;
                        }
                      } else {
                        result[0] += -0.05317132351386903;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.051747083663941318) ) ) {
                      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                        if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.593729019165039951) ) ) {
                          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.135134458541871005) ) ) {
                            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.29667711257934748) ) ) {
                              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
                                result[0] += -0.05340866508977345;
                              } else {
                                result[0] += 0.02578969779544793;
                              }
                            } else {
                              result[0] += 0.19364088436857355;
                            }
                          } else {
                            result[0] += -0.020374276705424382;
                          }
                        } else {
                          if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.665046453475953037) ) ) {
                            result[0] += -0.14643774132943882;
                          } else {
                            result[0] += -0.02247317267839609;
                          }
                        }
                      } else {
                        result[0] += -0.03543079916764687;
                      }
                    } else {
                      result[0] += -0.1279977835558676;
                    }
                  }
                } else {
                  result[0] += 0.0017949281301090005;
                }
              } else {
                result[0] += 0.014336898067099209;
              }
            }
          } else {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.426736354827881748) ) ) {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.515218973159790483) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.16594791412353693) ) ) {
                  if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.00030292528623755916;
                    } else {
                      result[0] += -0.01864288689101914;
                    }
                  } else {
                    if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.998158693313599077) ) ) {
                      result[0] += -0.0021695051909953393;
                    } else {
                      if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                        result[0] += 0.0006257013551624854;
                      } else {
                        if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.569529533386231357) ) ) {
                          if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)0.8958797454833985485) ) ) {
                            result[0] += 0.06564461147111277;
                          } else {
                            result[0] += -0.0663625419401584;
                          }
                        } else {
                          result[0] += -0.04375034202423597;
                        }
                      }
                    }
                  }
                } else {
                  result[0] += -0.014779376084176768;
                }
              } else {
                result[0] += -0.07518707527720134;
              }
            } else {
              result[0] += 0.008264690458245655;
            }
          }
        } else {
          result[0] += 0.0021396879060598473;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
      result[0] += 0.005731550977254683;
    } else {
      result[0] += -0.0002486956881328934;
    }
  } else {
    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
      if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.029068946838379794) ) ) {
            result[0] += -0.006241109623947527;
          } else {
            result[0] += -0.03328628698705964;
          }
        } else {
          if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.497866153717041238) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.1260108947753924) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += -0.02900023985007861;
              } else {
                result[0] += -0.0013918337218044954;
              }
            } else {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.011313159122951862;
                } else {
                  result[0] += -0.03963108117160283;
                }
              } else {
                if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)8816427008.000001907) ) ) {
                    result[0] += -0.11518132353286033;
                  } else {
                    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.005317695716673656;
                    } else {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.531673669815064365) ) ) {
                        result[0] += -0.032123668064273456;
                      } else {
                        result[0] += -0.0032873959959572535;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.12629127502441584) ) ) {
                    result[0] += -0.15070889734702933;
                  } else {
                    result[0] += -0.012855385006905196;
                  }
                }
              }
            }
          } else {
            result[0] += -0.03534610196763074;
          }
        }
      } else {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.028861761093140537) ) ) {
          if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
            if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)2.012675821781158891) ) ) {
              result[0] += 0.002417949339515718;
            } else {
              result[0] += 0.034884537399888164;
            }
          } else {
            if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)9.622284412384034979) ) ) {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.046861171722413886) ) ) {
                    if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += -0.05886725147603378;
                      } else {
                        result[0] += -0.015350142405400488;
                      }
                    } else {
                      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.777633190155030185) ) ) {
                        result[0] += 0.0105084427852383;
                      } else {
                        result[0] += -0.0456194931978198;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)4.531673669815064365) ) ) {
                      result[0] += 0.002645891464820421;
                    } else {
                      result[0] += 0.11760060696173091;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.835998296737671787) ) ) {
                    if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.658699750900269443) ) ) {
                      if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += 0.01833646536075253;
                      } else {
                        result[0] += -0.015140622261679943;
                      }
                    } else {
                      result[0] += 0.11969520423279648;
                    }
                  } else {
                    if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.637949228286744052) ) ) {
                      result[0] += -0.025520742194871063;
                    } else {
                      if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
                        result[0] += -0.1176701749968014;
                      } else {
                        result[0] += -0.04628730412781299;
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)1.497866153717041238) ) ) {
                  if ( UNLIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.011346421168716583;
                  } else {
                    result[0] += -0.1097282114666022;
                  }
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.497866153717041238) ) ) {
                    if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)3.000000000000000444) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.918693304061890537) ) ) {
                        result[0] += 0.6653634963455839;
                      } else {
                        result[0] += 0.14733087237442463;
                      }
                    } else {
                      result[0] += 0.03001239083988132;
                    }
                  } else {
                    if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += 0.010566622462087366;
                    } else {
                      result[0] += -0.013217625589316717;
                    }
                  }
                }
              }
            } else {
              result[0] += 0.010529739952176547;
            }
          }
        } else {
          result[0] += -0.0153070928446538;
        }
      }
    } else {
      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.888826131820679155) ) ) {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.770631790161133257) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.909254074096680576) ) ) {
              result[0] += 0.0025310625545185293;
            } else {
              result[0] += -0.07710587592545687;
            }
          } else {
            if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.540854334831238237) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.142630577087403232) ) ) {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += 0.03256549002908667;
                } else {
                  result[0] += -0.009952927937395944;
                }
              } else {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.09398412704467951) ) ) {
                    result[0] += -0.07174639439932073;
                  } else {
                    result[0] += 0.027903332762973545;
                  }
                } else {
                  result[0] += 0.11272182517100025;
                }
              }
            } else {
              result[0] += 0.049070934291786335;
            }
          }
        } else {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.51693725585937678) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.553712725639343706) ) ) {
              result[0] += -0.012779254037116187;
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.745876312255860263) ) ) {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.357691764831543413) ) ) {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.322819471359253818) ) ) {
                      if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
                          result[0] += -0.01689448787559301;
                        } else {
                          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                            result[0] += -0.0001273729860820755;
                          } else {
                            result[0] += -0.05399945810712295;
                          }
                        }
                      } else {
                        result[0] += 0.01074527608383668;
                      }
                    } else {
                      if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.040285587310792792) ) ) {
                        result[0] += 0.011403097750368497;
                      } else {
                        result[0] += -0.003916688839575488;
                      }
                    }
                  } else {
                    result[0] += -0.002006722102017322;
                  }
                } else {
                  result[0] += -0.03266659228937059;
                }
              } else {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.132848501205445224) ) ) {
                    result[0] += -0.004586229578136996;
                  } else {
                    result[0] += -0.034987198901234355;
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
                    result[0] += -0.05923053118488726;
                  } else {
                    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                      result[0] += 0.008319252576629896;
                    } else {
                      if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += -0.011571813193937342;
                      } else {
                        result[0] += 0.02106064964586211;
                      }
                    }
                  }
                }
              }
            }
          } else {
            result[0] += -0.01614433536956936;
          }
        }
      } else {
        result[0] += 0.0006513680432280658;
      }
    }
  }
  if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
      result[0] += 0.005273523103690382;
    } else {
      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
            if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
              if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.00225336263381221;
              } else {
                result[0] += 0.08056416315903801;
              }
            } else {
              result[0] += -0.007828055479879093;
            }
          } else {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.531673669815064365) ) ) {
              result[0] += -0.10314842689694462;
            } else {
              result[0] += -0.017397044899979622;
            }
          }
        } else {
          result[0] += -0.06604133764690118;
        }
      } else {
        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
              if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.022284501903122703;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.637949228286744052) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.329314231872559482) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.397998809814454013) ) ) {
                      result[0] += -0.026860339078476195;
                    } else {
                      result[0] += 0.08361506905459964;
                    }
                  } else {
                    result[0] += -0.041055664125171;
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.255632162094117099) ) ) {
                    if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += -0.017891312877499874;
                    } else {
                      result[0] += -0.10668765827206098;
                    }
                  } else {
                    result[0] += -0.00019427395976785348;
                  }
                }
              }
            } else {
              result[0] += -0.006868431525505108;
            }
          } else {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2252.000000000000455) ) ) {
              result[0] += -0.06505389358244866;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.513969182968140537) ) ) {
                result[0] += -0.06480008765270634;
              } else {
                result[0] += 0.018642074614799914;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.158952236175537998) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.2121162414550799) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += 0.0110883845868816;
                } else {
                  result[0] += -0.02904324132416837;
                }
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.909855604171753818) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.029068946838379794) ) ) {
                    result[0] += -0.11352045292438653;
                  } else {
                    result[0] += -0.013418649500659791;
                  }
                } else {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.673553824424744096) ) ) {
                    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                      result[0] += -0.0930355232356351;
                    } else {
                      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
                        result[0] += 0.02691265245112508;
                      } else {
                        result[0] += -0.07267629459697621;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.198464870452881303) ) ) {
                      result[0] += 0.04849644275385389;
                    } else {
                      result[0] += -0.013237958843607442;
                    }
                  }
                }
              }
            } else {
              result[0] += 0.03982146596131387;
            }
          } else {
            if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)6.563149929046631748) ) ) {
                if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)6.185178756713868076) ) ) {
                  result[0] += 0.009490617438306433;
                } else {
                  if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.0019429558039142467;
                  } else {
                    result[0] += 0.06721931816709283;
                  }
                }
              } else {
                result[0] += -0.030269915609598054;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.654679536819458896) ) ) {
                result[0] += -0.04609042244270639;
              } else {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
                  if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                    result[0] += 0.057257524771960056;
                  } else {
                    result[0] += -0.032193061664060586;
                  }
                } else {
                  result[0] += 0.09531498012673939;
                }
              }
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)8816427008.000001907) ) ) {
      if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
        result[0] += -0.06454747194959552;
      } else {
        result[0] += 0.08052380349219905;
      }
    } else {
      if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.07465314865112482) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.48738741874694913) ) ) {
            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.068990230560303623) ) ) {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.88435244560241788) ) ) {
                  if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.003019768553400133;
                  } else {
                    if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.23832273483276456) ) ) {
                        result[0] += 0.0016144343895564681;
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.941167116165162021) ) ) {
                          result[0] += 0.004909062604793735;
                        } else {
                          result[0] += -0.036206283929873996;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                        result[0] += -0.007198473402557314;
                      } else {
                        result[0] += 0.016642148100963188;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += 0.01334869346906622;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.288254261016846591) ) ) {
                      result[0] += 0.02127438605576179;
                    } else {
                      result[0] += -0.016472782201360035;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)12.00000000000000178) ) ) {
                  result[0] += 0.004570566336979958;
                } else {
                  result[0] += 0.02793015777496924;
                }
              }
            } else {
              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.350240230560303178) ) ) {
                result[0] += -0.016548462297812275;
              } else {
                result[0] += 0.008020052000934346;
              }
            }
          } else {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.993164777755738193) ) ) {
                result[0] += -0.0035163234704639074;
              } else {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.039614086690067274;
                } else {
                  result[0] += -0.007046121963694831;
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.674522399902344638) ) ) {
                result[0] += 0.0015400579195598877;
              } else {
                if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)8.285748958587648261) ) ) {
                    result[0] += -0.009161675792820225;
                  } else {
                    result[0] += -0.10537515886100486;
                  }
                } else {
                  result[0] += 0.0006694478565703702;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.431901693344116655) ) ) {
            if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
              if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.009892363720176142;
              } else {
                result[0] += 0.051566375623938114;
              }
            } else {
              result[0] += -0.035642294491665824;
            }
          } else {
            result[0] += -0.009467883509306758;
          }
        }
      } else {
        result[0] += 0.00030653761328356896;
      }
    }
  }
  if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1.868834793567657693) ) ) {
      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.914472818374634233) ) ) {
          result[0] += 0.10505966830359692;
        } else {
          result[0] += 0.02048085893988977;
        }
      } else {
        result[0] += -0.03451891702098471;
      }
    } else {
      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.718933820724488193) ) ) {
          result[0] += 0.0038888631515188337;
        } else {
          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
            if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
                if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                  result[0] += 0.1360780645554557;
                } else {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.36986422538757413) ) ) {
                      if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.138333082199097124) ) ) {
                        result[0] += 0.006157134004557064;
                      } else {
                        result[0] += -0.07361295502973961;
                      }
                    } else {
                      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
                        result[0] += -0.0348977241486202;
                      } else {
                        result[0] += 0.015961938592256;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.067782521247864214) ) ) {
                      if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += 0.040761356260488116;
                      } else {
                        result[0] += 0.1473281031301082;
                      }
                    } else {
                      result[0] += -0.012637152740542485;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.36986422538757413) ) ) {
                    result[0] += 0.12730994092955333;
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.80060577392578303) ) ) {
                      if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.602003335952759233) ) ) {
                        result[0] += -0.08249913522359717;
                      } else {
                        result[0] += 0.0007144389708805484;
                      }
                    } else {
                      result[0] += -0.12168872943878517;
                    }
                  }
                } else {
                  result[0] += -0.13469860080165807;
                }
              }
            } else {
              if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += -0.0009679969382848873;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.97887301445007413) ) ) {
                      result[0] += -0.09241305564056498;
                    } else {
                      if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.700598716735840066) ) ) {
                        result[0] += 0.034732986446000105;
                      } else {
                        result[0] += -0.09217246266409439;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
                    result[0] += -0.03337557924500185;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.664408206939698154) ) ) {
                      result[0] += -0.08832199306476549;
                    } else {
                      result[0] += 0.01325313709570003;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.814126014709473544) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.184114694595337802) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.74845767021179288) ) ) {
                          result[0] += 0.11894877325629623;
                        } else {
                          result[0] += -0.0030702296374060635;
                        }
                      } else {
                        result[0] += -0.015084666720091237;
                      }
                    } else {
                      result[0] += -0.10290906368301761;
                    }
                  } else {
                    if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)6.170655488967896396) ) ) {
                      result[0] += -0.048949404233434385;
                    } else {
                      result[0] += -0.0010166966390248633;
                    }
                  }
                } else {
                  result[0] += 0.019451697535323137;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.597218394279480425) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.855006217956543857) ) ) {
                result[0] += -0.06821553090357341;
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.715336322784424716) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.664408206939698154) ) ) {
                    result[0] += -0.08067646231608178;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.347943067550660068) ) ) {
                      result[0] += 0.05136949958296381;
                    } else {
                      result[0] += -0.011716845438977106;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.617236852645874912) ) ) {
                    result[0] += 0.03779732290280796;
                  } else {
                    result[0] += -0.025082529198300486;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.321723937988282138) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.65754175186157404) ) ) {
                    if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)6.090177536010743076) ) ) {
                      result[0] += -0.0163408284379188;
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.10618352890014826) ) ) {
                        result[0] += 0.130582731100715;
                      } else {
                        result[0] += 0.02073012338555126;
                      }
                    }
                  } else {
                    result[0] += 0.10036908666380107;
                  }
                } else {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.639559984207154208) ) ) {
                    result[0] += 0.01117291331444287;
                  } else {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += -0.002491969192266476;
                    } else {
                      result[0] += 0.11396783698227349;
                    }
                  }
                }
              } else {
                result[0] += 0.04606836752918099;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.41263532638549982) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.029068946838379794) ) ) {
            result[0] += 0.015762735886635317;
          } else {
            if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.03420138359069913) ) ) {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.15100884437561124) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.637949228286744052) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.637949228286744052) ) ) {
                    result[0] += 0.015792223816831007;
                  } else {
                    result[0] += -0.046218510390431344;
                  }
                } else {
                  result[0] += 0.029661528001146215;
                }
              } else {
                if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.551017761230469638) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.088880300521851474) ) ) {
                    result[0] += -0.06376859716725841;
                  } else {
                    result[0] += 0.004134607600906322;
                  }
                } else {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.668133974075318271) ) ) {
                    result[0] += -0.28917455575590784;
                  } else {
                    result[0] += 0.04602047550361902;
                  }
                }
              }
            } else {
              result[0] += -0.11707973470415711;
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.9055976867675799) ) ) {
            result[0] += -0.1626916426769213;
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.73117923736572443) ) ) {
              result[0] += 0.010267117931396674;
            } else {
              result[0] += -0.10305706263944524;
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)8816427008.000001907) ) ) {
      if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
          if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.349750161170959917) ) ) {
            result[0] += -0.041885951270877755;
          } else {
            result[0] += -0.1502745064578405;
          }
        } else {
          result[0] += -0.02371141308823639;
        }
      } else {
        if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += -0.011585651312100896;
        } else {
          result[0] += 0.1807837057842138;
        }
      }
    } else {
      result[0] += -0.00022278119912330033;
    }
  }
}

