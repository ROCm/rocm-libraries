
#include "header.h"

void predict_unit2(union Entry* data, double* result) {
  unsigned int tmp;
  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)20.00000000000000355) ) ) {
    if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
      if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
        if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)12.50000000000000178) ) ) {
          result[0] += 0.0011527332966379158;
        } else {
          result[0] += -0.06896510520907405;
        }
      } else {
        if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
          result[0] += -0.037544675452235915;
        } else {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)120.0000000000000142) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.154959201812744585) ) ) {
              result[0] += -0.0034855100019909376;
            } else {
              result[0] += 0.06352866904853792;
            }
          } else {
            result[0] += -0.014845194763805414;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)2.500000000000000444) ) ) {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.339395284652710849) ) ) {
          result[0] += 0.006107262632612205;
        } else {
          result[0] += 0.05346336679655522;
        }
      } else {
        if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)16.50000000000000355) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.02604460716247603) ) ) {
            result[0] += -0.09044076481093187;
          } else {
            result[0] += 0.010964829270840686;
          }
        } else {
          result[0] += -0.06018967664057992;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.215408444404602495) ) ) {
        result[0] += -0.014930921666969089;
      } else {
        result[0] += -0.07709942214659667;
      }
    } else {
      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.736135363578796831) ) ) {
        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)104.0000000000000142) ) ) {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.394753932952881748) ) ) {
                  result[0] += -0.004446253547771694;
                } else {
                  result[0] += 0.04666499819935721;
                }
              } else {
                result[0] += -0.01159248600924097;
              }
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.481121778488159624) ) ) {
                result[0] += 0.017803342434805577;
              } else {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.827801465988160068) ) ) {
                    result[0] += -0.08524614736222726;
                  } else {
                    result[0] += 0.1072325545641628;
                  }
                } else {
                  result[0] += -0.010682141813098069;
                }
              }
            }
          } else {
            result[0] += 0.018989020391364398;
          }
        } else {
          result[0] += -0.05611456359026264;
        }
      } else {
        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
          if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)4.500000000000000888) ) ) {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                result[0] += -0.1387884107068473;
              } else {
                result[0] += -0.024619797090714053;
              }
            } else {
              result[0] += 0.043092532374329306;
            }
          } else {
            if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)104.0000000000000142) ) ) {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)72.00000000000001421) ) ) {
                  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.142630577087403232) ) ) {
                      result[0] += 0.03428227461306084;
                    } else {
                      result[0] += -0.058004587784888685;
                    }
                  } else {
                    if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                        result[0] += 0.06508665983171044;
                      } else {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                          result[0] += 0.05054830137462068;
                        } else {
                          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.249904870986938921) ) ) {
                            result[0] += 0.11506626214052657;
                          } else {
                            result[0] += -0.09637247113363495;
                          }
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.172047138214112216) ) ) {
                          result[0] += -0.0377621430308245;
                        } else {
                          result[0] += 0.0311408407706618;
                        }
                      } else {
                        result[0] += 0.031059285468733558;
                      }
                    }
                  }
                } else {
                  result[0] += 0.19077005967166283;
                }
              } else {
                result[0] += -0.038526897873042705;
              }
            } else {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
                  if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += -0.07867651583920855;
                    } else {
                      result[0] += 0.011408876031333635;
                    }
                  } else {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += -0.08051390669212517;
                    } else {
                      result[0] += 0.025812174234005808;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.445705175399781162) ) ) {
                        if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)7.000000000000000888) ) ) {
                          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.607751369476319248) ) ) {
                            result[0] += 0.10621138083454512;
                          } else {
                            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)120.0000000000000142) ) ) {
                              if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
                                result[0] += 0.05299917202285429;
                              } else {
                                result[0] += -0.004682718023968016;
                              }
                            } else {
                              if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                                result[0] += 0.03989777108590803;
                              } else {
                                result[0] += -0.06051842045073122;
                              }
                            }
                          }
                        } else {
                          result[0] += -0.04902830380409838;
                        }
                      } else {
                        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                          if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)5.000000000000000888) ) ) {
                            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.516936540603638583) ) ) {
                              result[0] += -0.0488041888772707;
                            } else {
                              result[0] += 0.045516856316692986;
                            }
                          } else {
                            result[0] += -0.05512781313444657;
                          }
                        } else {
                          if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)3.500000000000000444) ) ) {
                            result[0] += -0.05574245513065129;
                          } else {
                            if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.039720773696899636) ) ) {
                              result[0] += 0.047992970387589276;
                            } else {
                              result[0] += -0.20744103692064592;
                            }
                          }
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                        result[0] += 0.052645531983415594;
                      } else {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.827801465988160068) ) ) {
                          result[0] += 0.03915210110265874;
                        } else {
                          result[0] += -0.11712734536721142;
                        }
                      }
                    }
                  } else {
                    result[0] += -0.06287221289914784;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.094205617904663974) ) ) {
                  result[0] += 0.0371896563674482;
                } else {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.156774044036865678) ) ) {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.142630577087403232) ) ) {
                        result[0] += 0.018755949313251086;
                      } else {
                        result[0] += -0.10238412605551639;
                      }
                    } else {
                      result[0] += -0.014563986186818645;
                    }
                  } else {
                    result[0] += -0.1173588292973906;
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.861792564392090288) ) ) {
            result[0] += -0.014110458735990213;
          } else {
            result[0] += -0.05576572871430113;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)5.500000000000000888) ) ) {
    if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
        if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
            if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += -0.07090656185927195;
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.617236852645874912) ) ) {
                result[0] += 0.002650271906871687;
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.516936540603638583) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.667095184326172763) ) ) {
                    result[0] += -0.6741575061081315;
                  } else {
                    result[0] += -0.042862446619689125;
                  }
                } else {
                  result[0] += -0.08811613031504327;
                }
              }
            }
          } else {
            result[0] += 0.013489791304445754;
          }
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)6.500000000000000888) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.1822080612182635) ) ) {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)232.0000000000000284) ) ) {
                  result[0] += -0.0004205909467957156;
                } else {
                  if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                    result[0] += -0.08054464729035339;
                  } else {
                    result[0] += 0.05378629004566098;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.605039834976196733) ) ) {
                    result[0] += -0.10239222105006357;
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.154959201812744585) ) ) {
                      result[0] += -0.09173514203437581;
                    } else {
                      result[0] += 0.016028096779210202;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                      result[0] += -0.02198901864550315;
                    } else {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.605039834976196733) ) ) {
                        result[0] += -0.10593032469445558;
                      } else {
                        result[0] += -0.04466875922582278;
                      }
                    }
                  } else {
                    result[0] += 0.0037967848809142717;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)8.500000000000001776) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.778982400894165927) ) ) {
                  if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                    result[0] += 0.0033811276871994495;
                  } else {
                    result[0] += -0.06626603193344026;
                  }
                } else {
                  result[0] += 0.028799730508359378;
                }
              } else {
                if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)9.500000000000001776) ) ) {
                  if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                    result[0] += -0.10960957795640472;
                  } else {
                    result[0] += 0.035095692196612786;
                  }
                } else {
                  if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)11.50000000000000178) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                      result[0] += -0.022678176939646664;
                    } else {
                      result[0] += 0.03848307561914849;
                    }
                  } else {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.284418344497681552) ) ) {
                      result[0] += 0.00945487310752804;
                    } else {
                      result[0] += -0.05917114574528384;
                    }
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.484580039978028232) ) ) {
                if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                    if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                      result[0] += -0.003945070391019748;
                    } else {
                      result[0] += 0.0695536659542313;
                    }
                  } else {
                    result[0] += -0.030651418039006107;
                  }
                } else {
                  result[0] += 0.054371545121208925;
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.475347042083742011) ) ) {
                  result[0] += 0.012870018132942277;
                } else {
                  result[0] += -0.08478213859034396;
                }
              }
            } else {
              if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += -0.03818390594693254;
              } else {
                if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.07091415811994066;
                } else {
                  if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                    result[0] += 0.004216261667830295;
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.094205617904663974) ) ) {
                      result[0] += -0.06756559229235734;
                    } else {
                      if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                        result[0] += 0.0632633037808765;
                      } else {
                        result[0] += 0.014234568925147382;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.48918962478637873) ) ) {
          if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)6.000000000000000888) ) ) {
            result[0] += -0.06570915946650967;
          } else {
            if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.02998820382020316;
            } else {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)104.0000000000000142) ) ) {
                result[0] += 0.06022785855455126;
              } else {
                result[0] += 8.04578682201764e-05;
              }
            }
          }
        } else {
          result[0] += -0.10280695370628914;
        }
      }
    } else {
      if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.484580039978028232) ) ) {
          if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += -0.03817058473101083;
          } else {
            if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)3.500000000000000444) ) ) {
              if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                result[0] += 0.40356934623672625;
              } else {
                if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)12.50000000000000178) ) ) {
                  result[0] += -0.06967393447179568;
                } else {
                  result[0] += 0.14903948231981004;
                }
              }
            } else {
              result[0] += -0.0584046918217308;
            }
          }
        } else {
          result[0] += -0.08047679408880325;
        }
      } else {
        result[0] += -0.008965839579319927;
      }
    }
  } else {
    if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)15.50000000000000178) ) ) {
      if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)9.500000000000001776) ) ) {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
          result[0] += 0.08960443859934643;
        } else {
          if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += -0.0662094008665271;
          } else {
            result[0] += 0.04706161597758473;
          }
        }
      } else {
        if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)12.50000000000000178) ) ) {
          result[0] += 0.011556331172195414;
        } else {
          result[0] += -0.06281383267081274;
        }
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
        if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)13.50000000000000178) ) ) {
          if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5.830332040786744052) ) ) {
            result[0] += 0.08382009139584108;
          } else {
            result[0] += 0.016194547489141002;
          }
        } else {
          result[0] += -0.04015508470811124;
        }
      } else {
        if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
          result[0] += -0.04548920484765543;
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.780892848968506748) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.142747402191162998) ) ) {
              result[0] += 0.011007392376338992;
            } else {
              if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)31.50000000000000355) ) ) {
                result[0] += -0.056819964392901004;
              } else {
                result[0] += 0.04323618487790769;
              }
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.124530076980591708) ) ) {
              result[0] += -0.0202128768225552;
            } else {
              result[0] += -0.12014476287894132;
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)7.500000000000000888) ) ) {
      result[0] += 0.0037183141847919655;
    } else {
      if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)9.783419609069825995) ) ) {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.90474271774292081) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.938867926597595659) ) ) {
              result[0] += -0.044645882699738064;
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.357691764831543413) ) ) {
                result[0] += -0.06797896457094285;
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.422362327575684482) ) ) {
                  result[0] += 0.0632695931748005;
                } else {
                  result[0] += -0.010199115421320668;
                }
              }
            }
          } else {
            result[0] += -0.03565605545382324;
          }
        } else {
          result[0] += -0.09922871006434668;
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.718933820724488193) ) ) {
          if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += -0.08579839133267377;
          } else {
            result[0] += 0.0283724058652125;
          }
        } else {
          result[0] += -0.07048673544178106;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.516936540603638583) ) ) {
        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
            result[0] += -0.05379262077362001;
          } else {
            result[0] += 0.06990276022785723;
          }
        } else {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
            result[0] += -0.04726199180536639;
          } else {
            result[0] += -0.08801322070932109;
          }
        }
      } else {
        if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.700598716735840066) ) ) {
            result[0] += 0.10481012061559879;
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.208071470260621005) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.516936540603638583) ) ) {
                if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)80.00000000000001421) ) ) {
                    if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)26.50000000000000355) ) ) {
                      result[0] += -0.003737112438005653;
                    } else {
                      result[0] += -0.15823358781661664;
                    }
                  } else {
                    result[0] += -0.1366004548851251;
                  }
                } else {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)120.0000000000000142) ) ) {
                    result[0] += -0.04326404694606811;
                  } else {
                    result[0] += 0.06748306424294372;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.06502546412748983;
                } else {
                  result[0] += 0.05444365765056312;
                }
              }
            } else {
              result[0] += 0.03997901576074861;
            }
          }
        } else {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.718933820724488193) ) ) {
            if ( LIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.06348450671801885;
            } else {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)56.00000000000000711) ) ) {
                result[0] += 0.06653628208936829;
              } else {
                result[0] += -0.03476244919693642;
              }
            }
          } else {
            result[0] += 0.016730763108120997;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)6.500000000000000888) ) ) {
        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.7512402534484881) ) ) {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.718933820724488193) ) ) {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += 0.008244450360890448;
              } else {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.198464870452881303) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.516936540603638583) ) ) {
                    if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                      result[0] += -0.1427975556731487;
                    } else {
                      result[0] += -0.005829948234712529;
                    }
                  } else {
                    result[0] += -0.08255703153608505;
                  }
                } else {
                  result[0] += 0.02005585230130892;
                }
              }
            } else {
              result[0] += -0.06101779863608812;
            }
          } else {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.284418344497681552) ) ) {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                    result[0] += -0.0656725079897732;
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.142630577087403232) ) ) {
                      result[0] += -0.022363159642561147;
                    } else {
                      result[0] += 0.04583239870723409;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.585059762001038042) ) ) {
                    result[0] += -0.07261106700878456;
                  } else {
                    result[0] += 0.04160738771994447;
                  }
                }
              } else {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                    result[0] += 0.0008390380891204036;
                  } else {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                      result[0] += -0.08728614208494875;
                    } else {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.357691764831543413) ) ) {
                        result[0] += 0.10910823597856012;
                      } else {
                        result[0] += -0.018685199020340073;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)4.500000000000000888) ) ) {
                    result[0] += -0.013720713677138755;
                  } else {
                    result[0] += 0.05422155524333995;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.778982400894165927) ) ) {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                    result[0] += 0.03971574596744302;
                  } else {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
                      result[0] += -0.035146892978675716;
                    } else {
                      result[0] += 0.01675598061481292;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)4.500000000000000888) ) ) {
                    result[0] += 0.010775608809004703;
                  } else {
                    result[0] += 0.049980590372827524;
                  }
                }
              } else {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                    if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2565.000000000000455) ) ) {
                      result[0] += -0.0685174930063314;
                    } else {
                      result[0] += 0.03849663099369753;
                    }
                  } else {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)72.00000000000001421) ) ) {
                      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                        result[0] += 0.0026737453924418303;
                      } else {
                        result[0] += -0.03744722838865687;
                      }
                    } else {
                      result[0] += 0.042370888440921226;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)2.500000000000000444) ) ) {
                      result[0] += 0.0074843594580971924;
                    } else {
                      result[0] += -0.1273057078958915;
                    }
                  } else {
                    result[0] += -0.10746678206044243;
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
            result[0] += -0.0867687171866372;
          } else {
            result[0] += 0.07542745705887942;
          }
        }
      } else {
        if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)9.500000000000001776) ) ) {
          result[0] += 0.039536081472107215;
        } else {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
            result[0] += -0.009146817385440128;
          } else {
            if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
              result[0] += 0.011624548859355897;
            } else {
              result[0] += 0.07394250959799704;
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
    if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)16.50000000000000355) ) ) {
      if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.79840612411499201) ) ) {
          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
            result[0] += 0.003850063964117122;
          } else {
            result[0] += -0.010156145698503996;
          }
        } else {
          if ( LIKELY( !(data[51].missing != -1) || (data[51].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.004643432669111588;
            } else {
              result[0] += 0.04494271441346596;
            }
          } else {
            result[0] += 0.09902233380381498;
          }
        }
      } else {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.516936540603638583) ) ) {
            if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.215905904769898349) ) ) {
                  result[0] += -0.0836802408807684;
                } else {
                  result[0] += 0.033937873799618115;
                }
              } else {
                if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)80.00000000000001421) ) ) {
                    result[0] += 0.06991461041496211;
                  } else {
                    result[0] += -0.0752632238451647;
                  }
                } else {
                  result[0] += -0.09208556692943806;
                }
              }
            } else {
              result[0] += -0.08679151943531005;
            }
          } else {
            if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.718933820724488193) ) ) {
                  result[0] += -0.07335060104463222;
                } else {
                  result[0] += -0.004607539562369633;
                }
              } else {
                result[0] += 0.015991022192042376;
              }
            } else {
              if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                result[0] += 0.005244894649629621;
              } else {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.445705175399781162) ) ) {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.012675821781158891) ) ) {
                    result[0] += 0.07714637303942351;
                  } else {
                    result[0] += -0.07865741973440561;
                  }
                } else {
                  result[0] += 0.13045919051188712;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.7512402534484881) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
                result[0] += 0.0028776114073212366;
              } else {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.032220827137280854;
                  } else {
                    result[0] += -0.06402626915973508;
                  }
                } else {
                  result[0] += -0.11344303451798186;
                }
              }
            } else {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.694163918495178667) ) ) {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += -0.06982357212452427;
                  } else {
                    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                      result[0] += 0.030171143261049423;
                    } else {
                      if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5.283100605010987216) ) ) {
                        if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)6.000000000000000888) ) ) {
                          if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.500000000000000444) ) ) {
                            result[0] += 0.017164277365357344;
                          } else {
                            result[0] += -0.06263387659870194;
                          }
                        } else {
                          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                            result[0] += -0.057132331558999855;
                          } else {
                            result[0] += -0.01170437371263701;
                          }
                        }
                      } else {
                        result[0] += -0.07546116896646006;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)47227863040.00000763) ) ) {
                    if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.44140100479126021) ) ) {
                      result[0] += 0.0058491684827558475;
                    } else {
                      result[0] += -0.10162668852077544;
                    }
                  } else {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.938867926597595659) ) ) {
                        result[0] += 0.07218122233534051;
                      } else {
                        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                          if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)137422176256.0000153) ) ) {
                            result[0] += -0.00988741042639574;
                          } else {
                            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.467917680740357333) ) ) {
                              result[0] += -0.06464139220931546;
                            } else {
                              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)120.0000000000000142) ) ) {
                                if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
                                  result[0] += -0.0625962122662674;
                                } else {
                                  result[0] += 0.010443747483949313;
                                }
                              } else {
                                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                                  result[0] += -0.07984298268719903;
                                } else {
                                  result[0] += 0.0612442543870675;
                                }
                              }
                            }
                          }
                        } else {
                          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)104.0000000000000142) ) ) {
                            result[0] += -0.017781800490893514;
                          } else {
                            result[0] += 0.04626067706696477;
                          }
                        }
                      }
                    } else {
                      result[0] += 0.024214328460717346;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.284418344497681552) ) ) {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.04539423938964826;
                  } else {
                    result[0] += 0.008841778637340326;
                  }
                } else {
                  if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                    result[0] += -0.08471860760461436;
                  } else {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += -0.02009855600407324;
                      } else {
                        result[0] += 0.01295513151873008;
                      }
                    } else {
                      if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                          result[0] += 0.09989542413706709;
                        } else {
                          result[0] += -0.06945541261688719;
                        }
                      } else {
                        result[0] += -0.11314152757159956;
                      }
                    }
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
              result[0] += -0.0844607499647133;
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.445705175399781162) ) ) {
                result[0] += -0.06579792113014736;
              } else {
                result[0] += 0.07850303851587828;
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.827801465988160068) ) ) {
        if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)10.50000000000000178) ) ) {
          result[0] += 0.04945862326105378;
        } else {
          result[0] += -0.020315760787922646;
        }
      } else {
        if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
          if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)13.50000000000000178) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.778982400894165927) ) ) {
              result[0] += 0.08529836093123971;
            } else {
              result[0] += -0.10616965072480454;
            }
          } else {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.215905904769898349) ) ) {
              result[0] += 0.022357471843351916;
            } else {
              result[0] += -0.07402422862202487;
            }
          }
        } else {
          result[0] += 0.022891835264762996;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.48918962478637873) ) ) {
      if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
        result[0] += -0.03584636448123968;
      } else {
        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)104.0000000000000142) ) ) {
          result[0] += 0.053913620738494865;
        } else {
          result[0] += -0.008178558677502595;
        }
      }
    } else {
      result[0] += -0.10133844452252999;
    }
  }
  if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
    if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)5.500000000000000888) ) ) {
      if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
          if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)240501391360.0000305) ) ) {
              if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2415.000000000000455) ) ) {
                if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
                    if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.031064545729654332;
                    } else {
                      result[0] += -0.029490079640340772;
                    }
                  } else {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.357691764831543413) ) ) {
                      result[0] += -0.13769616207726065;
                    } else {
                      result[0] += 0.03281645741239405;
                    }
                  }
                } else {
                  result[0] += 0.006334401450725018;
                }
              } else {
                result[0] += 0.06827304792412624;
              }
            } else {
              if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.497866153717041238) ) ) {
                  result[0] += 0.016832423854641034;
                } else {
                  result[0] += 0.07439737915270558;
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.361115694046021396) ) ) {
                  if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                    result[0] += -0.17543894792082482;
                  } else {
                    result[0] += -0.02785393566112944;
                  }
                } else {
                  result[0] += 0.019207109387088056;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)6.500000000000000888) ) ) {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.1822080612182635) ) ) {
                  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += -0.028204079115702375;
                  } else {
                    result[0] += 0.0025912663471204574;
                  }
                } else {
                  result[0] += -0.021873141809064803;
                }
              } else {
                if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.029068946838379794) ) ) {
                    if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.008868493379303162;
                    } else {
                      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)40.00000000000000711) ) ) {
                        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.802696108818054643) ) ) {
                          result[0] += 0.05267770578637137;
                        } else {
                          result[0] += -0.008077260329950929;
                        }
                      } else {
                        result[0] += 0.0006878058245729921;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)56.00000000000000711) ) ) {
                      result[0] += -0.0808543904470955;
                    } else {
                      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)20.00000000000000355) ) ) {
                            result[0] += -0.003129976121870677;
                          } else {
                            result[0] += -0.07650845501630128;
                          }
                        } else {
                          result[0] += 0.026529118177157508;
                        }
                      } else {
                        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.208071470260621005) ) ) {
                            if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                              result[0] += -0.03152409458811994;
                            } else {
                              result[0] += 0.037747679895432894;
                            }
                          } else {
                            result[0] += -0.06314640102359288;
                          }
                        } else {
                          result[0] += -0.07608155164175247;
                        }
                      }
                    }
                  }
                } else {
                  result[0] += 0.02744619517011624;
                }
              }
            } else {
              if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)8.500000000000001776) ) ) {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                  result[0] += 0.0401186227795754;
                } else {
                  if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                    result[0] += 0.01994201092117949;
                  } else {
                    result[0] += -0.039037392413919784;
                  }
                }
              } else {
                result[0] += -0.011582419177607935;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.029068946838379794) ) ) {
            result[0] += -0.07796493513391008;
          } else {
            if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += 0.09451902084652529;
            } else {
              result[0] += 0.020770897560651286;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.94957673549652144) ) ) {
          result[0] += -0.009387346226148046;
        } else {
          if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)4.500000000000000888) ) ) {
            if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)2.500000000000000444) ) ) {
              result[0] += -0.06872257882714775;
            } else {
              result[0] += -0.023908592292128907;
            }
          } else {
            result[0] += 0.026260092134651948;
          }
        }
      }
    } else {
      if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.478159427642823154) ) ) {
          if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)9.500000000000001776) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.90474271774292081) ) ) {
              result[0] += 0.03914760162596057;
            } else {
              result[0] += -0.029517583238024576;
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.284418344497681552) ) ) {
              result[0] += -0.07783887434979124;
            } else {
              if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)10.50000000000000178) ) ) {
                result[0] += -0.10792956121674716;
              } else {
                result[0] += 0.02285728895490452;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.780892848968506748) ) ) {
            if ( LIKELY( !(data[51].missing != -1) || (data[51].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += 0.020956851843344212;
            } else {
              result[0] += -0.09190458958818104;
            }
          } else {
            result[0] += -0.16898934788750164;
          }
        }
      } else {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
          if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.743881702423096591) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.623839378356934482) ) ) {
                result[0] += -0.07775124359691865;
              } else {
                result[0] += 0.009185240774951666;
              }
            } else {
              result[0] += 0.012350708478679072;
            }
          } else {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.88435244560241788) ) ) {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.208071470260621005) ) ) {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.070054531097412998) ) ) {
                    if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)10.50000000000000178) ) ) {
                      result[0] += -0.009459366308826335;
                    } else {
                      result[0] += -0.14769207001429627;
                    }
                  } else {
                    result[0] += 0.05387060665773166;
                  }
                } else {
                  result[0] += 0.034557594663087936;
                }
              } else {
                if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)7.500000000000000888) ) ) {
                  result[0] += 0.15062669163993342;
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.743881702423096591) ) ) {
                    result[0] += 0.03535741778256451;
                  } else {
                    result[0] += -0.07038116548838246;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)6.500000000000000888) ) ) {
                  result[0] += -0.08963291518170273;
                } else {
                  result[0] += 0.05488226815553372;
                }
              } else {
                if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)31.50000000000000355) ) ) {
                  result[0] += -0.10086949339010132;
                } else {
                  result[0] += -0.00758449761223399;
                }
              }
            }
          }
        } else {
          result[0] += 0.007783395628090422;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.48918962478637873) ) ) {
      result[0] += 0.03254520506761204;
    } else {
      result[0] += -0.10053910608783445;
    }
  }
  if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
    if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)11.50000000000000178) ) ) {
        if ( UNLIKELY(  (data[42].missing != -1) && (data[42].fvalue <= (double)-1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)240501391360.0000305) ) ) {
            result[0] += 0.0001696793893955095;
          } else {
            if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)9.500000000000001776) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.055836200714113104) ) ) {
                result[0] += 0.0032743176510732237;
              } else {
                result[0] += 0.04003552817785075;
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.744781017303467685) ) ) {
                if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)11.50000000000000178) ) ) {
                  result[0] += -0.05663785764739637;
                } else {
                  result[0] += 0.01586566492679325;
                }
              } else {
                result[0] += -0.05012804368816148;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.888826131820679155) ) ) {
              result[0] += -0.015522496477435963;
            } else {
              result[0] += -0.0936812115745053;
            }
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.605039834976196733) ) ) {
                result[0] += -0.09112845036491636;
              } else {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.039286790517648856;
                  } else {
                    result[0] += 0.004440155283094216;
                  }
                } else {
                  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += -0.03087674382435628;
                  } else {
                    result[0] += 0.0008951618473103109;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.029068946838379794) ) ) {
                    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)56.00000000000000711) ) ) {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.802696108818054643) ) ) {
                        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.142630577087403232) ) ) {
                            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                              result[0] += 0.04385390017419066;
                            } else {
                              result[0] += -0.04025273459356002;
                            }
                          } else {
                            result[0] += 0.07025886933044326;
                          }
                        } else {
                          result[0] += 0.09255085091164877;
                        }
                      } else {
                        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.778982400894165927) ) ) {
                            result[0] += 0.019494808070895122;
                          } else {
                            result[0] += -0.06329428055068263;
                          }
                        } else {
                          result[0] += -0.060207042643796164;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                        result[0] += -0.031167254653229216;
                      } else {
                        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.142630577087403232) ) ) {
                            if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.764309883117677558) ) ) {
                                  result[0] += -0.02664258944265199;
                                } else {
                                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.142630577087403232) ) ) {
                                    result[0] += 0.03194672425186062;
                                  } else {
                                    result[0] += -0.043907043154561544;
                                  }
                                }
                              } else {
                                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.671854496002199042) ) ) {
                                  result[0] += 0.01446074394002617;
                                } else {
                                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.516936540603638583) ) ) {
                                      result[0] += -0.06801368449399774;
                                    } else {
                                      result[0] += 0.05670220981333401;
                                    }
                                  } else {
                                    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                                      result[0] += 0.0623364952221464;
                                    } else {
                                      if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                                        result[0] += -0.0347992597344115;
                                      } else {
                                        result[0] += 0.0021980336761742232;
                                      }
                                    }
                                  }
                                }
                              }
                            } else {
                              if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
                                result[0] += -0.011637001061701754;
                              } else {
                                result[0] += -0.048421844163145704;
                              }
                            }
                          } else {
                            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)80.00000000000001421) ) ) {
                              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.516936540603638583) ) ) {
                                result[0] += 0.012039475768624873;
                              } else {
                                if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
                                  result[0] += -0.1321530729620103;
                                } else {
                                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.673553824424744096) ) ) {
                                    result[0] += -0.004546129915773138;
                                  } else {
                                    if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                                      result[0] += -0.09142442778396476;
                                    } else {
                                      result[0] += -0.019161334152270202;
                                    }
                                  }
                                }
                              }
                            } else {
                              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.484580039978028232) ) ) {
                                result[0] += -0.03603216288698328;
                              } else {
                                result[0] += 0.03319197159735556;
                              }
                            }
                          }
                        } else {
                          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.142630577087403232) ) ) {
                            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.978769779205324042) ) ) {
                              result[0] += -0.05022838671530433;
                            } else {
                              result[0] += 0.027951532866773827;
                            }
                          } else {
                            result[0] += -0.03294352160167375;
                          }
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)56.00000000000000711) ) ) {
                      result[0] += -0.07721274423924145;
                    } else {
                      if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                        result[0] += -0.028739239655799234;
                      } else {
                        if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
                          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                            result[0] += -0.0807825493815365;
                          } else {
                            result[0] += 0.0018949482265699865;
                          }
                        } else {
                          if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                            result[0] += 0.008394867832674624;
                          } else {
                            result[0] += 0.06501951088311683;
                          }
                        }
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += 0.0013142336289279986;
                  } else {
                    result[0] += 0.04257666689525991;
                  }
                }
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.029068946838379794) ) ) {
                  result[0] += -0.08715773865513193;
                } else {
                  result[0] += 0.07557997138658228;
                }
              }
            }
          }
        }
      } else {
        result[0] += -0.053389917997461595;
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.744781017303467685) ) ) {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.516936540603638583) ) ) {
          result[0] += -0.07076851662295501;
        } else {
          result[0] += 0.008173612829635753;
        }
      } else {
        if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
          result[0] += 0.025309861183671206;
        } else {
          if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
            result[0] += -0.037591427046956234;
          } else {
            if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)16.50000000000000355) ) ) {
              result[0] += 0.03393177752546927;
            } else {
              result[0] += -0.10270213349787037;
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.938867926597595659) ) ) {
      result[0] += -0.0283377481636925;
    } else {
      if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
        result[0] += -0.03507547684305997;
      } else {
        if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)6.000000000000000888) ) ) {
          result[0] += -0.08970247866094469;
        } else {
          result[0] += 0.04408298937069887;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)5.500000000000000888) ) ) {
      if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)10.50000000000000178) ) ) {
        if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)6.500000000000000888) ) ) {
          if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)2.500000000000000444) ) ) {
            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( LIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.055836200714113104) ) ) {
                  if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.607751369476319248) ) ) {
                      result[0] += -0.008649466230062432;
                    } else {
                      result[0] += 0.038249782714817805;
                    }
                  } else {
                    result[0] += -0.10243839076909074;
                  }
                } else {
                  result[0] += 0.008989978746333494;
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.07729864120483576) ) ) {
                  result[0] += 0.026086261588840573;
                } else {
                  result[0] += -0.0467298511622538;
                }
              }
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.6749763488769549) ) ) {
                if ( LIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.01249438470821871;
                } else {
                  result[0] += 0.06366839959267834;
                }
              } else {
                if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)112.0000000000000142) ) ) {
                  result[0] += -0.0056579323182508285;
                } else {
                  result[0] += 0.06729942721716017;
                }
              }
            }
          } else {
            result[0] += -0.06275505760391283;
          }
        } else {
          if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)7.500000000000000888) ) ) {
            if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
              result[0] += 0.10860880647976112;
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.124530076980591708) ) ) {
                result[0] += -0.07158210514522358;
              } else {
                result[0] += 0.10030677450999857;
              }
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.95478391647339045) ) ) {
              result[0] += 0.027799697403417103;
            } else {
              result[0] += -0.1292635428238643;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += -0.08880789058204061;
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.094205617904663974) ) ) {
            result[0] += 0.034604701958833105;
          } else {
            result[0] += -0.06366031595746673;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.215905904769898349) ) ) {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.361115694046021396) ) ) {
          if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)9.500000000000001776) ) ) {
            result[0] += 0.04529941347560484;
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.284418344497681552) ) ) {
              result[0] += -0.07113097218648587;
            } else {
              result[0] += 0.032078143625106456;
            }
          }
        } else {
          if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)6.500000000000000888) ) ) {
            result[0] += -0.06052262318952803;
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.481347560882569248) ) ) {
              result[0] += 0.036054962678918154;
            } else {
              result[0] += -0.046322570907750424;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.780892848968506748) ) ) {
          if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)6.500000000000000888) ) ) {
            result[0] += -0.10133389008392135;
          } else {
            result[0] += 0.003822138484110041;
          }
        } else {
          result[0] += -0.15873392994901897;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.516936540603638583) ) ) {
        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
          if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
            if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += -0.07138994273169838;
            } else {
              if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.09229031121221176;
              } else {
                result[0] += -0.04185777086447349;
              }
            }
          } else {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.142747402191162998) ) ) {
              result[0] += -0.08819263711517855;
            } else {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.04387167730732364;
              } else {
                result[0] += 0.16643074705893365;
              }
            }
          }
        } else {
          result[0] += -0.08534674395021176;
        }
      } else {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.718933820724488193) ) ) {
          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += -0.06529214356615208;
            } else {
              result[0] += 0.0051084577414599345;
            }
          } else {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += 0.00330406670032331;
            } else {
              result[0] += 0.10274910828237337;
            }
          }
        } else {
          if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
            result[0] += 0.00464720404926121;
          } else {
            result[0] += 0.07587217438228196;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.04636812210083185) ) ) {
        if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
          if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)4.500000000000000888) ) ) {
              if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.004132102142518734;
                } else {
                  result[0] += 0.013172205560376344;
                }
              } else {
                if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                    result[0] += -0.04387792240264779;
                  } else {
                    result[0] += 0.08092447104585426;
                  }
                } else {
                  result[0] += -0.005949008556419944;
                }
              }
            } else {
              if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.718933820724488193) ) ) {
                  result[0] += 0.0021729494234958174;
                } else {
                  result[0] += -0.07320570087786453;
                }
              } else {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.516936540603638583) ) ) {
                    result[0] += -0.07471539696631072;
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
                      result[0] += -0.015727580025464;
                    } else {
                      result[0] += 0.08959491691697588;
                    }
                  }
                } else {
                  result[0] += 0.012182017478155796;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.467917680740357333) ) ) {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += -0.10597664119398134;
                } else {
                  result[0] += 0.006523130354672792;
                }
              } else {
                result[0] += 0.014006352157645617;
              }
            } else {
              result[0] += 0.029068704249569033;
            }
          }
        } else {
          result[0] += -0.01689249077875603;
        }
      } else {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += -0.07971720732031963;
            } else {
              result[0] += 0.04006750080729948;
            }
          } else {
            result[0] += -0.0673725073661318;
          }
        } else {
          if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += 0.05415741676795232;
            } else {
              result[0] += -0.06960895761945708;
            }
          } else {
            result[0] += -0.0021552500742590373;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
    if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)11.50000000000000178) ) ) {
        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
          if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.042435407638550693) ) ) {
                      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += 0.00033541213415733806;
                      } else {
                        result[0] += -0.08767916588995145;
                      }
                    } else {
                      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                        if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)5.500000000000000888) ) ) {
                          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                            result[0] += 0.021133184342946452;
                          } else {
                            result[0] += 0.06536794253752057;
                          }
                        } else {
                          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.780892848968506748) ) ) {
                            result[0] += 0.00974923949931993;
                          } else {
                            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.124530076980591708) ) ) {
                              result[0] += 0.028940920218469803;
                            } else {
                              result[0] += -0.12182864796755466;
                            }
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += 0.007149795962573886;
                        } else {
                          result[0] += -0.04986037696621562;
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.03056481141198545;
                    } else {
                      result[0] += -0.09541755459711486;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.484580039978028232) ) ) {
                    result[0] += -0.16702615318660416;
                  } else {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.255632162094117099) ) ) {
                      result[0] += 0.0192548891809474;
                    } else {
                      result[0] += -0.14435926904155597;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.993164777755738193) ) ) {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.379217386245728427) ) ) {
                    result[0] += 0.011412229689393854;
                  } else {
                    result[0] += -0.01068743803130073;
                  }
                } else {
                  if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)5.500000000000000888) ) ) {
                    if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.04844477383922214;
                    } else {
                      if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.500000000000000444) ) ) {
                        result[0] += -0.07923259247157952;
                      } else {
                        if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                          result[0] += 0.01606263844315684;
                        } else {
                          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                            result[0] += 0.0026297696309707467;
                          } else {
                            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
                              result[0] += -0.10506464096009833;
                            } else {
                              result[0] += -0.023670184367902247;
                            }
                          }
                        }
                      }
                    }
                  } else {
                    result[0] += -0.05560388862121168;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.284418344497681552) ) ) {
                  result[0] += -0.0037971286868746855;
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.962127923965454546) ) ) {
                    result[0] += 0.03675528587383952;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.87254357337951749) ) ) {
                      result[0] += 0.03704698333377925;
                    } else {
                      result[0] += -0.04685002857142337;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.802696108818054643) ) ) {
                  result[0] += -0.05286995496731688;
                } else {
                  result[0] += 0.03115491761375888;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.888826131820679155) ) ) {
                result[0] += -0.016551528586632287;
              } else {
                result[0] += -0.09220468010243969;
              }
            } else {
              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.605039834976196733) ) ) {
                  if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.019885022262741053;
                  } else {
                    result[0] += -0.09532668044124636;
                  }
                } else {
                  if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                        result[0] += -0.0435598437025315;
                      } else {
                        result[0] += -0.00484975076295045;
                      }
                    } else {
                      result[0] += 0.00043803570107677305;
                    }
                  } else {
                    if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += -0.059576392933907985;
                      } else {
                        result[0] += -0.015528589760737517;
                      }
                    } else {
                      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)168.0000000000000284) ) ) {
                        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.605039834976196733) ) ) {
                          result[0] += -0.05048356972855093;
                        } else {
                          result[0] += 0.003785233350625355;
                        }
                      } else {
                        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.780892848968506748) ) ) {
                          result[0] += -0.06564774223403574;
                        } else {
                          result[0] += 0.07841652089741348;
                        }
                      }
                    }
                  }
                }
              } else {
                result[0] += 0.0002758982515637871;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
            result[0] += -0.07730892191445118;
          } else {
            if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += 0.08445169612665628;
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
                result[0] += -0.04691671078497103;
              } else {
                result[0] += 0.042397493138014275;
              }
            }
          }
        }
      } else {
        result[0] += -0.05019058184884308;
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
        result[0] += -0.06187944337023972;
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.694163918495178667) ) ) {
          result[0] += -0.03996837664691132;
        } else {
          if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)2.500000000000000444) ) ) {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.534971714019776279) ) ) {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.05565302186002024;
              } else {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.141444921493531162) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.433569431304932529) ) ) {
                      result[0] += -0.1604625629402713;
                    } else {
                      result[0] += -0.05178299776452397;
                    }
                  } else {
                    result[0] += 0.014099845845209983;
                  }
                } else {
                  result[0] += 0.03388407998339301;
                }
              }
            } else {
              result[0] += 0.05034745944448883;
            }
          } else {
            result[0] += 0.009537325048461208;
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)104.0000000000000142) ) ) {
      if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
        result[0] += -0.03654755246942467;
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.154959201812744585) ) ) {
          result[0] += -0.014181271367809188;
        } else {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.947818994522095615) ) ) {
            result[0] += 0.05887487002356489;
          } else {
            result[0] += -0.02586412389587743;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.284418344497681552) ) ) {
        result[0] += -0.08691774861461737;
      } else {
        result[0] += 0.0018343893701461408;
      }
    }
  }
  if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
    if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)16.50000000000000355) ) ) {
      if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
        if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)240501391360.0000305) ) ) {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
              result[0] += 0.03296816792616013;
            } else {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.038949824841107955;
              } else {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                    result[0] += 0.0036159168377487894;
                  } else {
                    result[0] += -0.025070746482890857;
                  }
                } else {
                  result[0] += 0.022250148114782292;
                }
              }
            }
          } else {
            result[0] += 0.024281467989139605;
          }
        } else {
          if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)12.50000000000000178) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.827801465988160068) ) ) {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[51].missing != -1) || (data[51].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)31.50000000000000355) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.35709619522094904) ) ) {
                        result[0] += -0.014418548102698948;
                      } else {
                        result[0] += 0.009336154214764312;
                      }
                    } else {
                      result[0] += -0.20234446674991396;
                    }
                  } else {
                    result[0] += 0.05960683946686737;
                  }
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.48918962478637873) ) ) {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                      if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.869339942932130683) ) ) {
                          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                            result[0] += 0.030770044477430387;
                          } else {
                            result[0] += 0.2393351972927659;
                          }
                        } else {
                          result[0] += -0.016295571266674472;
                        }
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.834949493408204901) ) ) {
                          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                            result[0] += -0.027569284529482027;
                          } else {
                            result[0] += 0.091295444922801;
                          }
                        } else {
                          result[0] += -0.07591471342120612;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.060294389724732333) ) ) {
                        result[0] += 0.009731358847144937;
                      } else {
                        if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)1.500000000000000222) ) ) {
                          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                              result[0] += -0.030078918887532655;
                            } else {
                              result[0] += 0.06708178097759786;
                            }
                          } else {
                            if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
                              result[0] += 0.0988190661195861;
                            } else {
                              if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                                result[0] += 0.08623882664981569;
                              } else {
                                result[0] += -0.005435772206711796;
                              }
                            }
                          }
                        } else {
                          result[0] += 0.020618109014952774;
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += -0.08120056670557049;
                    } else {
                      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                        result[0] += 0.033367413298173505;
                      } else {
                        result[0] += -0.05087277291374566;
                      }
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.039720773696899636) ) ) {
                  if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.669892311096192294) ) ) {
                      result[0] += -0.06797011665993545;
                    } else {
                      if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
                        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)4.500000000000000888) ) ) {
                          result[0] += 0.019675384965993734;
                        } else {
                          result[0] += 0.04892224274212061;
                        }
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.10290670394897639) ) ) {
                          result[0] += -0.08662237775789883;
                        } else {
                          result[0] += 0.008870879549222837;
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
                      if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5.283100605010987216) ) ) {
                        result[0] += -0.03512563113841227;
                      } else {
                        result[0] += 0.06058743846752012;
                      }
                    } else {
                      if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)20.00000000000000355) ) ) {
                        result[0] += 0.1670203269448719;
                      } else {
                        result[0] += -0.011349342795950972;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.602003335952759233) ) ) {
                    result[0] += -0.09853290375158251;
                  } else {
                    result[0] += -0.020311671989932;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)7.500000000000000888) ) ) {
                  if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)6.500000000000000888) ) ) {
                    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)56.00000000000000711) ) ) {
                      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
                        if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)2.500000000000000444) ) ) {
                            result[0] += 0.018328271154360257;
                          } else {
                            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.284418344497681552) ) ) {
                              result[0] += 0.06665137293497711;
                            } else {
                              result[0] += -0.047420001054627056;
                            }
                          }
                        } else {
                          result[0] += -0.053676704421261295;
                        }
                      } else {
                        result[0] += -0.08403329198375502;
                      }
                    } else {
                      if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += -0.030748016611577195;
                      } else {
                        result[0] += 0.0026963241444495113;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)8.500000000000001776) ) ) {
                      result[0] += 0.031075948148990347;
                    } else {
                      result[0] += -0.00043390475023189753;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.013165561009752653;
                  } else {
                    result[0] += -0.050712525037698034;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.198464870452881303) ) ) {
                  result[0] += -0.10183905742720188;
                } else {
                  if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                    result[0] += -0.07871882146727796;
                  } else {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)56.00000000000000711) ) ) {
                        result[0] += -0.024591641373736987;
                      } else {
                        result[0] += 0.012898353881012313;
                      }
                    } else {
                      if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                        result[0] += 0.020760989844532604;
                      } else {
                        result[0] += -0.11930915601986916;
                      }
                    }
                  }
                }
              }
            }
          } else {
            result[0] += -0.0605647064156432;
          }
        }
      } else {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.029068946838379794) ) ) {
          result[0] += -0.07670224217605383;
        } else {
          if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
            result[0] += 0.0813650499796682;
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
              result[0] += -0.04532088024986875;
            } else {
              result[0] += 0.03508003437689802;
            }
          }
        }
      }
    } else {
      result[0] += 0.011924349314440516;
    }
  } else {
    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)104.0000000000000142) ) ) {
      if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
        result[0] += -0.03667776928233373;
      } else {
        result[0] += 0.039637300422488204;
      }
    } else {
      result[0] += -0.01909537356577678;
    }
  }
  if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
    if ( LIKELY(  (data[42].missing != -1) && (data[42].fvalue <= (double)-1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.778982400894165927) ) ) {
          result[0] += -0.00044404308045430335;
        } else {
          result[0] += 0.01617469623236688;
        }
      } else {
        if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)11.50000000000000178) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
            if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)8.500000000000001776) ) ) {
              if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)2.500000000000000444) ) ) {
                  result[0] += -0.13232112171595942;
                } else {
                  if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)4.500000000000000888) ) ) {
                    result[0] += 0.12210753032685025;
                  } else {
                    result[0] += 0.02169488654618426;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                  result[0] += -0.034086569055699886;
                } else {
                  result[0] += -0.11627239683722762;
                }
              }
            } else {
              result[0] += 0.04958784947419881;
            }
          } else {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2202.000000000000455) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.605039834976196733) ) ) {
                if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)10.50000000000000178) ) ) {
                  result[0] += -0.1196771289761019;
                } else {
                  if ( LIKELY( !(data[51].missing != -1) || (data[51].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)17.50000000000000355) ) ) {
                      result[0] += 0.053928325684103044;
                    } else {
                      result[0] += -0.01487971768475887;
                    }
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.154959201812744585) ) ) {
                      result[0] += 0.05727677616383925;
                    } else {
                      result[0] += -0.08171333177637169;
                    }
                  }
                }
              } else {
                result[0] += -0.0591114211551843;
              }
            } else {
              if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)6.500000000000000888) ) ) {
                if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.918272972106934482) ) ) {
                    result[0] += -0.07069105193183732;
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
                      result[0] += 0.028967664008560552;
                    } else {
                      result[0] += 0.1317739458363339;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                    result[0] += 0.007571577089698707;
                  } else {
                    result[0] += -0.05545984139198306;
                  }
                }
              } else {
                if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)16.50000000000000355) ) ) {
                  result[0] += 0.016376254697330814;
                } else {
                  if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)21.50000000000000355) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.484580039978028232) ) ) {
                      if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)19.50000000000000355) ) ) {
                        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.827801465988160068) ) ) {
                          result[0] += 0.0160505869995353;
                        } else {
                          result[0] += -0.09933736637346878;
                        }
                      } else {
                        result[0] += 0.06521025131991585;
                      }
                    } else {
                      result[0] += -0.07099249597160251;
                    }
                  } else {
                    result[0] += 0.0033329590643887026;
                  }
                }
              }
            }
          }
        } else {
          result[0] += -0.0401801853761405;
        }
      }
    } else {
      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.740319490432739702) ) ) {
          if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.736135363578796831) ) ) {
                result[0] += 0.012687598289818153;
              } else {
                result[0] += -0.04794602182041973;
              }
            } else {
              if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                result[0] += 0.1890504334776122;
              } else {
                result[0] += -0.08873734401668851;
              }
            }
          } else {
            result[0] += 0.028059215720874706;
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
            result[0] += -0.011285277868104132;
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
              if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += -0.04452458450059937;
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.918272972106934482) ) ) {
                  result[0] += 0.11685166385597481;
                } else {
                  result[0] += -0.03403506539343582;
                }
              }
            } else {
              result[0] += -0.09268031311269116;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
          if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)240501391360.0000305) ) ) {
            result[0] += 0.006544550850127263;
          } else {
            result[0] += -0.09239877450368594;
          }
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.605039834976196733) ) ) {
              result[0] += -0.09682963160447633;
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.605039834976196733) ) ) {
                result[0] += -0.0554388481414745;
              } else {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)88.00000000000001421) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.90474271774292081) ) ) {
                    result[0] += 0.0034341635325182427;
                  } else {
                    result[0] += -0.02695433184116639;
                  }
                } else {
                  if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)240501391360.0000305) ) ) {
                    if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                      result[0] += 0.05563071146585727;
                    } else {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.339395284652710849) ) ) {
                        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += 0.024343565894230392;
                        } else {
                          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                            result[0] += -0.08841882509773796;
                          } else {
                            result[0] += 0.008266011853518164;
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                            result[0] += 0.07649890896266998;
                          } else {
                            result[0] += -0.06260560500712757;
                          }
                        } else {
                          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.43742394447326749) ) ) {
                            result[0] += -0.01862998318690594;
                          } else {
                            result[0] += 0.06887717409119243;
                          }
                        }
                      }
                    }
                  } else {
                    result[0] += -0.05799549820096792;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)6.000000000000000888) ) ) {
              if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.284418344497681552) ) ) {
                  result[0] += -0.0525158745311568;
                } else {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += 0.0746190984947303;
                  } else {
                    result[0] += -0.09308846738394813;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += 0.7816082761556188;
                } else {
                  result[0] += 0.03322097332350114;
                }
              }
            } else {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                result[0] += 0.003327998186398379;
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.029068946838379794) ) ) {
                  result[0] += -0.08226157125164536;
                } else {
                  if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += 0.09069503226259235;
                  } else {
                    result[0] += 0.03035972649250511;
                  }
                }
              }
            }
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)104.0000000000000142) ) ) {
      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
        result[0] += -0.007378322862641652;
      } else {
        result[0] += 0.04368602974009762;
      }
    } else {
      result[0] += -0.021115062899634607;
    }
  }
  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)20.00000000000000355) ) ) {
    if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
      if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
        if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
          if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)11.50000000000000178) ) ) {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)56.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.605039834976196733) ) ) {
                if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.0001225350771728888;
                  } else {
                    result[0] += 0.02215237492076428;
                  }
                } else {
                  result[0] += 0.04673588612243213;
                }
              } else {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                  result[0] += 0.002838704939177541;
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.02604460716247603) ) ) {
                    if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)18.50000000000000355) ) ) {
                      result[0] += -0.03170231492853278;
                    } else {
                      result[0] += 0.098028578057415;
                    }
                  } else {
                    result[0] += -0.05056535202924141;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.44140100479126021) ) ) {
                result[0] += -0.002034421878156548;
              } else {
                result[0] += -0.06870255206003634;
              }
            }
          } else {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.433569431304932529) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.607751369476319248) ) ) {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)11.50000000000000178) ) ) {
                      result[0] += -0.15327472990493798;
                    } else {
                      result[0] += -0.015519527314949377;
                    }
                  } else {
                    result[0] += -0.01083161367582419;
                  }
                } else {
                  result[0] += 0.007783753752327743;
                }
              } else {
                result[0] += 0.017314642151778022;
              }
            } else {
              result[0] += -0.05416051086244096;
            }
          }
        } else {
          if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)6.000000000000000888) ) ) {
            result[0] += -0.06857737092685935;
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.357691764831543413) ) ) {
              result[0] += -0.06867493322945127;
            } else {
              if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.994480729103088823) ) ) {
                  result[0] += 0.4641005049876741;
                } else {
                  result[0] += -0.04300228543764381;
                }
              } else {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)120.0000000000000142) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.743881702423096591) ) ) {
                    result[0] += 0.05193201603550627;
                  } else {
                    result[0] += -0.012845302567065512;
                  }
                } else {
                  result[0] += -0.012326020141361966;
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
          result[0] += -0.04928737765305674;
        } else {
          if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)2.500000000000000444) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.261864185333252841) ) ) {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.481347560882569248) ) ) {
                  result[0] += 0.07273552785287285;
                } else {
                  result[0] += -0.09171192652711792;
                }
              } else {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.90474271774292081) ) ) {
                    result[0] += -0.06693976422694756;
                  } else {
                    result[0] += 0.12047797104192524;
                  }
                } else {
                  result[0] += 0.026815313481233973;
                }
              }
            } else {
              result[0] += 0.07958722584426134;
            }
          } else {
            if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)16.50000000000000355) ) ) {
              result[0] += 0.010012500062843467;
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.920601367950440341) ) ) {
                result[0] += -0.0656629889489735;
              } else {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                  result[0] += -0.10092725976610147;
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.059293270111084873) ) ) {
                    if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)11.50000000000000178) ) ) {
                      result[0] += 4.66519516357981;
                    } else {
                      result[0] += 1.8611925096024322;
                    }
                  } else {
                    result[0] += -0.08676217527773258;
                  }
                }
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
        result[0] += -0.0713708489683228;
      } else {
        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
          result[0] += 0.07410205443044006;
        } else {
          result[0] += 0.01031084250014036;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.418141007423401323) ) ) {
        result[0] += -0.009603810506853256;
      } else {
        if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
          result[0] += -0.06583029751741962;
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.13987779617309748) ) ) {
            result[0] += 0.03901645461299361;
          } else {
            result[0] += -0.02821567614743689;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.909855604171753818) ) ) {
          if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)7.000000000000000888) ) ) {
            if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.888826131820679155) ) ) {
                result[0] += 0.00402374856198812;
              } else {
                result[0] += -0.021868150150034887;
              }
            } else {
              if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
                result[0] += 0.008817215523580172;
              } else {
                result[0] += 0.08045489456060549;
              }
            }
          } else {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.255632162094117099) ) ) {
                if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)112.0000000000000142) ) ) {
                  result[0] += -0.021090286089933255;
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.607751369476319248) ) ) {
                    result[0] += 0.023700162098060626;
                  } else {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.524927973747253862) ) ) {
                        result[0] += 0.039465778875941125;
                      } else {
                        result[0] += -0.024026727267158186;
                      }
                    } else {
                      result[0] += -0.07782080898314009;
                    }
                  }
                }
              } else {
                result[0] += -0.08237255879763689;
              }
            } else {
              result[0] += -0.09751134183663973;
            }
          }
        } else {
          result[0] += -0.03243312259081156;
        }
      } else {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.909855604171753818) ) ) {
          if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)21466447872.00000381) ) ) {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.715336322784424716) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                result[0] += -0.06788680388464031;
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.778982400894165927) ) ) {
                  result[0] += -0.03608129308304539;
                } else {
                  result[0] += 0.012041479364633836;
                }
              }
            } else {
              result[0] += 0.04814415375968583;
            }
          } else {
            result[0] += -0.06773132240002722;
          }
        } else {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.516936540603638583) ) ) {
              result[0] += -0.07114458945713073;
            } else {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                result[0] += 0.09446745431498582;
              } else {
                result[0] += -0.040109522045620204;
              }
            }
          } else {
            result[0] += -0.0698646413002809;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)20.00000000000000355) ) ) {
    if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
      if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
        if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)16.50000000000000355) ) ) {
          if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)240501391360.0000305) ) ) {
              if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                result[0] += 0.031443919488875115;
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.802696108818054643) ) ) {
                  if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.03314225305194336;
                  } else {
                    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                      result[0] += -0.003226671880932638;
                    } else {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.497866153717041238) ) ) {
                        result[0] += -0.015622745396924959;
                      } else {
                        result[0] += -0.05812510183012494;
                      }
                    }
                  }
                } else {
                  result[0] += 0.009529545983538735;
                }
              }
            } else {
              if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.497866153717041238) ) ) {
                  result[0] += 0.01624617762369632;
                } else {
                  result[0] += 0.06722928350247624;
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.361115694046021396) ) ) {
                  result[0] += -0.06575192934281875;
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.924581527709961826) ) ) {
                    result[0] += 0.023566551226543258;
                  } else {
                    result[0] += -0.1069150086311465;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)12.50000000000000178) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.651049375534058505) ) ) {
                if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)12.50000000000000178) ) ) {
                  if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                      if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += -0.03696986297724963;
                      } else {
                        if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.500000000000000444) ) ) {
                          result[0] += -0.0252274512145632;
                        } else {
                          result[0] += 0.010838165234364342;
                        }
                      }
                    } else {
                      result[0] += 0.006434521493000559;
                    }
                  } else {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += 0.045738015485174914;
                    } else {
                      result[0] += 0.009353826859347299;
                    }
                  }
                } else {
                  result[0] += -0.06230378403105472;
                }
              } else {
                if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)7.500000000000000888) ) ) {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += 0.0008979012120051972;
                  } else {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.449861526489258257) ) ) {
                      result[0] += -0.09275831134903084;
                    } else {
                      if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += -0.008509082274638266;
                      } else {
                        result[0] += -0.08594626998239968;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)6.500000000000000888) ) ) {
                    result[0] += -0.07072905362138046;
                  } else {
                    result[0] += -0.01130844982918839;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.962127923965454546) ) ) {
                result[0] += 0.09237900706879598;
              } else {
                result[0] += -0.0803252827382069;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.154959201812744585) ) ) {
            if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)19.50000000000000355) ) ) {
              result[0] += -0.08554993970920069;
            } else {
              result[0] += 0.00838946264083535;
            }
          } else {
            if ( UNLIKELY( !(data[51].missing != -1) || (data[51].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.0011846494159902705;
            } else {
              if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)10.50000000000000178) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
                  result[0] += 0.10951482372493773;
                } else {
                  result[0] += 0.03698203291467472;
                }
              } else {
                result[0] += 0.012086810177613774;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)6.000000000000000888) ) ) {
          result[0] += -0.06493522299602648;
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.357691764831543413) ) ) {
            result[0] += -0.06376563030309532;
          } else {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.42478513717651456) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.094205617904663974) ) ) {
                result[0] += 0.0005179935504138533;
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.43742394447326749) ) ) {
                  result[0] += 0.056630492999386654;
                } else {
                  result[0] += -0.03621996172198843;
                }
              }
            } else {
              result[0] += -0.009268652240407906;
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
        result[0] += -0.06929982218687493;
      } else {
        if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
          result[0] += 0.0681388701684724;
        } else {
          result[0] += 0.009079768760722034;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)56.00000000000000711) ) ) {
      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
        if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.827801465988160068) ) ) {
            if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2415.000000000000455) ) ) {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.888826131820679155) ) ) {
                  result[0] += -0.015141189813041207;
                } else {
                  result[0] += -0.08571376760287036;
                }
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
                  result[0] += 0.16943365971048352;
                } else {
                  result[0] += -0.0868934015290716;
                }
              }
            } else {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                result[0] += -0.09552607287244921;
              } else {
                result[0] += 0.004117907018978575;
              }
            }
          } else {
            if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
              result[0] += -0.08488293980537473;
            } else {
              result[0] += -0.019112375962698453;
            }
          }
        } else {
          result[0] += 0.0538314271734561;
        }
      } else {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.131699204444885698) ) ) {
          result[0] += -0.020290703609956797;
        } else {
          result[0] += -0.05647327757638664;
        }
      }
    } else {
      if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)7.000000000000000888) ) ) {
        if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.888826131820679155) ) ) {
            if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
              result[0] += -0.14277920956953416;
            } else {
              result[0] += 0.00332575417087213;
            }
          } else {
            if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.500000000000000444) ) ) {
              result[0] += -0.04322470292101256;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.41211462020874201) ) ) {
                result[0] += -0.06085597568421526;
              } else {
                result[0] += 0.0023991163285102825;
              }
            }
          }
        } else {
          result[0] += 0.004985923225110362;
        }
      } else {
        if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
          result[0] += -0.09987646082304025;
        } else {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.124530076980591708) ) ) {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += 0.014769828586735421;
              } else {
                result[0] += 0.15126055909491457;
              }
            } else {
              result[0] += -0.010046489283746154;
            }
          } else {
            result[0] += -0.05951560222170219;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)20.00000000000000355) ) ) {
    if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
      if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
        if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
          if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)9.500000000000001776) ) ) {
            if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.0004047245620091118;
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
                result[0] += 0.09469978768037854;
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.342454433441162998) ) ) {
                  result[0] += 0.04751792861113149;
                } else {
                  result[0] += -0.03045358871177277;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.736135363578796831) ) ) {
                result[0] += 0.05288923177668731;
              } else {
                result[0] += -0.07307999389880915;
              }
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.142630577087403232) ) ) {
                if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)6.500000000000000888) ) ) {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.0392545199424614;
                  } else {
                    result[0] += 0.004975972082994799;
                  }
                } else {
                  result[0] += -0.0694403223715838;
                }
              } else {
                if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)15.50000000000000178) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.993164777755738193) ) ) {
                    if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                      if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)11.50000000000000178) ) ) {
                        if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)3.500000000000000444) ) ) {
                          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                            result[0] += 0.3067213016126411;
                          } else {
                            result[0] += 0.9395185872395976;
                          }
                        } else {
                          result[0] += 0.055005579556892295;
                        }
                      } else {
                        result[0] += -0.0832930701806144;
                      }
                    } else {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.357691764831543413) ) ) {
                        result[0] += 0.003612197392597895;
                      } else {
                        result[0] += -0.0595293559672353;
                      }
                    }
                  } else {
                    result[0] += -0.076999153889593;
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.94957673549652144) ) ) {
                    if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.07612467474043765;
                    } else {
                      result[0] += 0.05944078357994645;
                    }
                  } else {
                    if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.013165827643695636;
                    } else {
                      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                        result[0] += -0.0011801383805867633;
                      } else {
                        result[0] += -0.07766915421064735;
                      }
                    }
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)120.0000000000000142) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.743881702423096591) ) ) {
              result[0] += 0.03365537418370037;
            } else {
              result[0] += -0.027461887573103562;
            }
          } else {
            result[0] += -0.025897823987966535;
          }
        }
      } else {
        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.938867926597595659) ) ) {
          result[0] += -0.07554340809267938;
        } else {
          if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)2.500000000000000444) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.261864185333252841) ) ) {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.05951567348952887;
              } else {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.043311427045269635;
                } else {
                  result[0] += 0.023505879847885797;
                }
              }
            } else {
              result[0] += 0.07505762515502067;
            }
          } else {
            if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)16.50000000000000355) ) ) {
              result[0] += 0.008631753615131852;
            } else {
              result[0] += -0.05263015796580133;
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
        result[0] += -0.06727973299828915;
      } else {
        if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
          result[0] += 0.06198178400067366;
        } else {
          result[0] += 0.008209794478665154;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.418141007423401323) ) ) {
        result[0] += -0.008507281776285435;
      } else {
        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
          result[0] += -0.06401410305154752;
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.13987779617309748) ) ) {
            result[0] += 0.03534803936446883;
          } else {
            result[0] += -0.02767729464886696;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.849175214767456943) ) ) {
          if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)7.000000000000000888) ) ) {
            if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.87548160552978693) ) ) {
                    result[0] += 0.012881777314977111;
                  } else {
                    result[0] += 0.06549477241233372;
                  }
                } else {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.868834793567657693) ) ) {
                    result[0] += 0.11738250156713492;
                  } else {
                    result[0] += -0.08392226553389498;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)56.00000000000000711) ) ) {
                    if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)112.0000000000000142) ) ) {
                      result[0] += -0.11684769546412288;
                    } else {
                      result[0] += 0.04248815473319254;
                    }
                  } else {
                    result[0] += -0.07256993599554694;
                  }
                } else {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)104.0000000000000142) ) ) {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.48918962478637873) ) ) {
                        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.516936540603638583) ) ) {
                            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.040716171264650214) ) ) {
                              result[0] += 0.02334540757285194;
                            } else {
                              result[0] += -0.049975181171980146;
                            }
                          } else {
                            result[0] += -9.767141447142197e-05;
                          }
                        } else {
                          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.154959201812744585) ) ) {
                            result[0] += -0.05656481044060675;
                          } else {
                            if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.802696108818054643) ) ) {
                              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.394753932952881748) ) ) {
                                result[0] += 0.03992143010961688;
                              } else {
                                result[0] += 0.0033867430917862134;
                              }
                            } else {
                              result[0] += -0.021710796469852866;
                            }
                          }
                        }
                      } else {
                        result[0] += -0.047482566406126483;
                      }
                    } else {
                      result[0] += 0.01665387990121691;
                    }
                  } else {
                    result[0] += 0.033714358634700424;
                  }
                }
              }
            } else {
              result[0] += 0.07637791443822323;
            }
          } else {
            if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
              result[0] += -0.09872968421035225;
            } else {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.255632162094117099) ) ) {
                result[0] += -0.00575440568599132;
              } else {
                result[0] += -0.07541346714396317;
              }
            }
          }
        } else {
          result[0] += -0.030742865462328562;
        }
      } else {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.909855604171753818) ) ) {
          result[0] += -0.03454275040933329;
        } else {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += 0.051109477687459284;
          } else {
            result[0] += -0.06351890153914573;
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)6.500000000000000888) ) ) {
    if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)20.00000000000000355) ) ) {
      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)1.994492053985595925) ) ) {
        if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += 0.0021441016127695884;
        } else {
          result[0] += -0.085561299109517;
        }
      } else {
        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
          if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
              if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)3072.000000000000455) ) ) {
                result[0] += 0.09657437975043887;
              } else {
                result[0] += 0.015065907294750103;
              }
            } else {
              result[0] += 0.0008003672079888702;
            }
          } else {
            result[0] += -0.00216920421917099;
          }
        } else {
          if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
            result[0] += 0.052625920711884294;
          } else {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
              result[0] += -0.0988498718286803;
            } else {
              result[0] += 0.008056703249847265;
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)56.00000000000000711) ) ) {
        if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.44140100479126021) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.30829524993896662) ) ) {
            if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.861792564392090288) ) ) {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.06881197310945575;
                } else {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += 0.038290799346443444;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.040716171264650214) ) ) {
                      result[0] += 0.4813760026589051;
                    } else {
                      result[0] += 0.11972960884829847;
                    }
                  }
                }
              } else {
                result[0] += -0.017867359507240475;
              }
            } else {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                  result[0] += 0.042283030002233386;
                } else {
                  result[0] += -0.037771035919188206;
                }
              } else {
                result[0] += -0.004075496532190651;
              }
            }
          } else {
            if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.94957673549652144) ) ) {
                  result[0] += 0.022982701513092892;
                } else {
                  result[0] += -0.03844904230072444;
                }
              } else {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.379217386245728427) ) ) {
                    result[0] += -0.07054217723427955;
                  } else {
                    result[0] += -0.0020940390744154914;
                  }
                } else {
                  result[0] += -0.01902900283106465;
                }
              }
            } else {
              if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)4.500000000000000888) ) ) {
                result[0] += -0.06492956236607354;
              } else {
                result[0] += 0.09018113115605259;
              }
            }
          }
        } else {
          result[0] += 0.0850216115421724;
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.02604460716247603) ) ) {
          if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
            result[0] += -0.10939617790134215;
          } else {
            result[0] += 0.007708488946889865;
          }
        } else {
          if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.700598716735840066) ) ) {
              result[0] += 0.03799478246021292;
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.97438240051269709) ) ) {
                  result[0] += -0.09255702836292407;
                } else {
                  result[0] += -0.016323220295444175;
                }
              } else {
                result[0] += -0.07345019027901942;
              }
            }
          } else {
            if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)7.000000000000000888) ) ) {
              if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.03624959454516942;
                } else {
                  result[0] += 0.01022300533664262;
                }
              } else {
                result[0] += -0.027783489004774127;
              }
            } else {
              if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += -0.09910243284146114;
              } else {
                result[0] += -0.011001294774865828;
              }
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)9.500000000000001776) ) ) {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
        result[0] += 0.07312643818010245;
      } else {
        if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
            result[0] += 0.06932113150046186;
          } else {
            result[0] += -0.07960849716284348;
          }
        } else {
          result[0] += 0.03322775192669405;
        }
      }
    } else {
      if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.516936540603638583) ) ) {
            result[0] += -0.07032801692596748;
          } else {
            result[0] += 0.015477220034401865;
          }
        } else {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.154959201812744585) ) ) {
              result[0] += -0.05247949299996198;
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.920601367950440341) ) ) {
                if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.970608234405518466) ) ) {
                  result[0] += 0.048127584081904456;
                } else {
                  result[0] += -0.08178605626579238;
                }
              } else {
                result[0] += -0.050469563325813704;
              }
            }
          } else {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.56941866874694913) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.623839378356934482) ) ) {
                  result[0] += -0.08950064546907555;
                } else {
                  result[0] += 0.018567638777102376;
                }
              } else {
                result[0] += 0.009683588258718581;
              }
            } else {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                result[0] += 0.012093003242830176;
              } else {
                result[0] += 0.04385197424298397;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.718933820724488193) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.361115694046021396) ) ) {
            if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.04551753869026553;
            } else {
              if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)13.50000000000000178) ) ) {
                result[0] += -0.014619589932088981;
              } else {
                result[0] += 0.018662551103573937;
              }
            }
          } else {
            result[0] += -0.04069374298527395;
          }
        } else {
          if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)17.50000000000000355) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.484580039978028232) ) ) {
              if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += 0.1993131500723363;
                } else {
                  result[0] += -0.011284653781714678;
                }
              } else {
                result[0] += -0.04431273899371798;
              }
            } else {
              result[0] += -0.07327687274411075;
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.070054531097412998) ) ) {
              result[0] += 4.425468128686839e-05;
            } else {
              if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.920601367950440341) ) ) {
                  result[0] += 1.4367253295107902;
                } else {
                  result[0] += 0.2924072366848806;
                }
              } else {
                result[0] += -0.0644633188911347;
              }
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)4.500000000000000888) ) ) {
        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.743881702423096591) ) ) {
            result[0] += 0.0006064393717623659;
          } else {
            if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)2.500000000000000444) ) ) {
                  result[0] += 0.04323581299087537;
                } else {
                  result[0] += -0.009442770687028978;
                }
              } else {
                result[0] += -0.03529676646146814;
              }
            } else {
              result[0] += -0.08266527001907276;
            }
          }
        } else {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
            result[0] += -0.06330234905988227;
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.623839378356934482) ) ) {
                result[0] += -0.030420779499045564;
              } else {
                result[0] += 0.061111863509578104;
              }
            } else {
              result[0] += 0.04817547322605978;
            }
          }
        }
      } else {
        result[0] += 0.09524953844606354;
      }
    } else {
      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)112.0000000000000142) ) ) {
        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.802696108818054643) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.184516429901124823) ) ) {
            result[0] += 0.03949271623817646;
          } else {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
              if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
                if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)21466447872.00000381) ) ) {
                  result[0] += -0.004508929495999361;
                } else {
                  result[0] += 0.12440559268482763;
                }
              } else {
                result[0] += -0.027596850228546844;
              }
            } else {
              result[0] += 0.0004955380124915354;
            }
          }
        } else {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)72.00000000000001421) ) ) {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += -0.0718721922890466;
            } else {
              result[0] += -0.03130878756231783;
            }
          } else {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.012675821781158891) ) ) {
              result[0] += -0.06891120014480832;
            } else {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)168.0000000000000284) ) ) {
                if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                  result[0] += 0.07771081943544171;
                } else {
                  result[0] += 0.002907864497135636;
                }
              } else {
                result[0] += -0.11125340135307807;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
          if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.484580039978028232) ) ) {
              if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.067782521247864214) ) ) {
                  result[0] += -0.09960444382954854;
                } else {
                  result[0] += -0.021419672622345495;
                }
              } else {
                if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)2.2821741104125981) ) ) {
                  if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
                      result[0] += 0.06925238252421045;
                    } else {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.02604460716247603) ) ) {
                        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                          result[0] += -0.020526637019215255;
                        } else {
                          result[0] += 0.00800352243859229;
                        }
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.1822080612182635) ) ) {
                          result[0] += 0.07670834703089241;
                        } else {
                          result[0] += -0.05313605360520775;
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += 0.03031733303868178;
                    } else {
                      result[0] += -0.009727446110401166;
                    }
                  }
                } else {
                  result[0] += 0.07024714000204656;
                }
              }
            } else {
              if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.040716171264650214) ) ) {
                  result[0] += 0.09384772472353073;
                } else {
                  if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                    result[0] += -0.028535410504518945;
                  } else {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += -0.08513734804836724;
                    } else {
                      result[0] += -0.007295070798298527;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.607751369476319248) ) ) {
                    result[0] += -0.012676438828795622;
                  } else {
                    result[0] += -0.05775176116206884;
                  }
                } else {
                  if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                    if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)7.000000000000000888) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.142630577087403232) ) ) {
                        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.131699204444885698) ) ) {
                          result[0] += -0.1689205281774427;
                        } else {
                          result[0] += -0.008658775926550401;
                        }
                      } else {
                        result[0] += 0.04816628434329241;
                      }
                    } else {
                      result[0] += -0.10734114247873228;
                    }
                  } else {
                    if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.718933820724488193) ) ) {
                        result[0] += 0.01315736213777456;
                      } else {
                        result[0] += -0.05288132897173743;
                      }
                    } else {
                      result[0] += -0.02255019903183577;
                    }
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
              result[0] += 0.003217558184899106;
            } else {
              result[0] += 0.05819188710334087;
            }
          }
        } else {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.088880300521851474) ) ) {
            if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
              result[0] += -0.07013089279492864;
            } else {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.778982400894165927) ) ) {
                result[0] += -0.0362973115323725;
              } else {
                if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
                  result[0] += -0.1168724053530367;
                } else {
                  result[0] += 0.020430675500140983;
                }
              }
            }
          } else {
            result[0] += 0.03270347145300103;
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.744781017303467685) ) ) {
      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.172047138214112216) ) ) {
        if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.484580039978028232) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.02604460716247603) ) ) {
              result[0] += -0.12168848289312251;
            } else {
              result[0] += 0.018051846206202222;
            }
          } else {
            result[0] += -0.21264965089577223;
          }
        } else {
          result[0] += -0.09505669218164864;
        }
      } else {
        result[0] += 0.00752659696340911;
      }
    } else {
      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.342454433441162998) ) ) {
        if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
          result[0] += 0.013718524993132981;
        } else {
          result[0] += -0.04787577317964725;
        }
      } else {
        if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)2.500000000000000444) ) ) {
          result[0] += 0.03200553922990193;
        } else {
          if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)16.50000000000000355) ) ) {
            result[0] += 0.012879845275579147;
          } else {
            if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)240501391360.0000305) ) ) {
                result[0] += 0.08208779208676076;
              } else {
                result[0] += -0.10335433529743106;
              }
            } else {
              result[0] += -0.11395928959376085;
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)7.500000000000000888) ) ) {
      result[0] += 0.001211976517593324;
    } else {
      if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.970608234405518466) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.481347560882569248) ) ) {
            result[0] += 0.03877806772306001;
          } else {
            result[0] += -0.05937397331539496;
          }
        } else {
          result[0] += -0.08307451003621595;
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.744781017303467685) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.361115694046021396) ) ) {
            result[0] += 0.0333431314323555;
          } else {
            result[0] += -0.026885549370053642;
          }
        } else {
          result[0] += -0.04188296234346145;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
      if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.615554332733154741) ) ) {
        result[0] += 0.06699561350948341;
      } else {
        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
            result[0] += -0.027922816534104597;
          } else {
            result[0] += 0.03208283095243633;
          }
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.246492147445679599) ) ) {
            result[0] += -0.08933554383668399;
          } else {
            result[0] += 0.04993847946421721;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.516936540603638583) ) ) {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.384830474853516513) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.599987030029298651) ) ) {
                  result[0] += 0.2577113023036634;
                } else {
                  result[0] += -0.0679534891895128;
                }
              } else {
                result[0] += 0.10802926908440763;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.95235633850097834) ) ) {
                result[0] += 0.0808068189684971;
              } else {
                if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)6.500000000000000888) ) ) {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.284418344497681552) ) ) {
                      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                        result[0] += -0.056734629516998675;
                      } else {
                        result[0] += 0.01416202092727604;
                      }
                    } else {
                      result[0] += 0.012932309387072728;
                    }
                  } else {
                    result[0] += 0.051339752452535505;
                  }
                } else {
                  result[0] += 0.0840997605422827;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.040716171264650214) ) ) {
              result[0] += 0.030488470489120362;
            } else {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.07650351055373288;
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.172047138214112216) ) ) {
                    if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.0440686584278173;
                    } else {
                      result[0] += -0.14958172751389712;
                    }
                  } else {
                    result[0] += -0.008724681559438967;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += -0.07201782855586214;
                } else {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += -0.07141551547408263;
                  } else {
                    if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.500000000000000444) ) ) {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.764703154563904253) ) ) {
                        result[0] += 0.02794881948983366;
                      } else {
                        result[0] += -0.021036998486053966;
                      }
                    } else {
                      result[0] += 0.054600498947291576;
                    }
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.516936540603638583) ) ) {
            if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)137422176256.0000153) ) ) {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)29.50000000000000355) ) ) {
                  result[0] += 0.008991234852861634;
                } else {
                  result[0] += 0.05216093042998228;
                }
              } else {
                result[0] += -0.014749581814426871;
              }
            } else {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                result[0] += -0.0815120883690937;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.623839378356934482) ) ) {
                  result[0] += -0.04258101036703226;
                } else {
                  result[0] += 0.002944673811970495;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)18.50000000000000355) ) ) {
              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.994492053985595925) ) ) {
                result[0] += -0.06084555782418816;
              } else {
                result[0] += 0.01581918076825741;
              }
            } else {
              if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)31.50000000000000355) ) ) {
                result[0] += -0.07874613548591863;
              } else {
                result[0] += 0.06749674495396384;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.516936540603638583) ) ) {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.38936424255371271) ) ) {
              result[0] += -0.13588171183168313;
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
                result[0] += 0.03212378460778729;
              } else {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += -0.006272119738229014;
                } else {
                  result[0] += -0.05927158613845749;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.56849193572998225) ) ) {
                result[0] += 0.005655128766748819;
              } else {
                result[0] += -0.04553804997491366;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.669892311096192294) ) ) {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                  result[0] += 0.1260698207496488;
                } else {
                  result[0] += -0.06507919825921259;
                }
              } else {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += 0.034913959292498055;
                  } else {
                    if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                      result[0] += 0.043686773573686305;
                    } else {
                      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)6.000000000000000888) ) ) {
                        result[0] += -0.07445718066643663;
                      } else {
                        result[0] += 0.0032460397458591445;
                      }
                    }
                  }
                } else {
                  result[0] += 0.011583714737669271;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.44140100479126021) ) ) {
            result[0] += -0.08653410928766625;
          } else {
            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
                if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)15.50000000000000178) ) ) {
                  if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                    result[0] += -0.011990236355586097;
                  } else {
                    result[0] += -0.06138785003788101;
                  }
                } else {
                  result[0] += 0.02109159104593592;
                }
              } else {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.607751369476319248) ) ) {
                  result[0] += 0.04392231682274925;
                } else {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)120.0000000000000142) ) ) {
                    result[0] += -0.044349207034506344;
                  } else {
                    result[0] += 0.02894391227011355;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += 0.012466678334789802;
              } else {
                result[0] += -0.095307366200027;
              }
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
      if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)11.50000000000000178) ) ) {
        if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)112.0000000000000142) ) ) {
                  result[0] += -0.0033315669243798515;
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.208071470260621005) ) ) {
                    result[0] += 0.05619202291473999;
                  } else {
                    result[0] += -0.05853921555908457;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.607751369476319248) ) ) {
                  result[0] += -0.06665156961088729;
                } else {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += -0.060935880642625055;
                  } else {
                    result[0] += 0.0091795397275145;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.994492053985595925) ) ) {
                result[0] += -0.0007296081506238732;
              } else {
                result[0] += 0.013280549906665499;
              }
            }
          } else {
            if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.114721298217775214) ) ) {
                result[0] += -0.06825133522206922;
              } else {
                if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)9.500000000000001776) ) ) {
                  if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)6.500000000000000888) ) ) {
                    result[0] += 0.006775761503219672;
                  } else {
                    result[0] += 0.0606574797255863;
                  }
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.361115694046021396) ) ) {
                    if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)13.50000000000000178) ) ) {
                      result[0] += -0.047567302228971695;
                    } else {
                      result[0] += -0.0002556448151470755;
                    }
                  } else {
                    result[0] += -0.06966920401236383;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.827801465988160068) ) ) {
                result[0] += 0.017322919093303725;
              } else {
                if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)10.50000000000000178) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.628996372222901279) ) ) {
                    result[0] += -0.0674171122807121;
                  } else {
                    result[0] += 0.11365233579736084;
                  }
                } else {
                  result[0] += -0.01158138694981805;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.605039834976196733) ) ) {
              result[0] += -0.08518817995773104;
            } else {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += -0.05101820345182924;
                  } else {
                    result[0] += -0.004210869970270296;
                  }
                } else {
                  result[0] += 0.0011487389031193992;
                }
              } else {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)88.00000000000001421) ) ) {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += -0.09273556777302969;
                  } else {
                    result[0] += 0.0005310240332770487;
                  }
                } else {
                  if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)240501391360.0000305) ) ) {
                    result[0] += -0.004584159701442286;
                  } else {
                    result[0] += -0.05362325681564637;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.484580039978028232) ) ) {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)56.00000000000000711) ) ) {
                  if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.500000000000000444) ) ) {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.215408444404602495) ) ) {
                      result[0] += -0.007647931951466765;
                    } else {
                      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                        result[0] += 0.0021862359015628586;
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.184516429901124823) ) ) {
                          result[0] += 0.0708096837237847;
                        } else {
                          result[0] += -0.0899155640642459;
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.827801465988160068) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.97438240051269709) ) ) {
                        if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += 0.06945265783118508;
                        } else {
                          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)112.0000000000000142) ) ) {
                            result[0] += 0.06654464011955703;
                          } else {
                            result[0] += -0.07304208481931222;
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.312486410140991655) ) ) {
                          result[0] += 0.053147986685552706;
                        } else {
                          result[0] += -0.03919942758327763;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                        result[0] += 0.06661800510596388;
                      } else {
                        result[0] += 0.304356948723989;
                      }
                    }
                  }
                } else {
                  result[0] += 0.005404687821541853;
                }
              } else {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.700598716735840066) ) ) {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)120.0000000000000142) ) ) {
                      result[0] += 0.01827587657273979;
                    } else {
                      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                        result[0] += 0.2297794922753104;
                      } else {
                        result[0] += -0.05884846855376902;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.516936540603638583) ) ) {
                      result[0] += -0.008651054434114961;
                    } else {
                      result[0] += -0.09634535673297628;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.08862828619302013;
                    } else {
                      if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += -0.027064571100851715;
                      } else {
                        if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += -0.13576325756290894;
                        } else {
                          if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                            result[0] += 0.08620916799875401;
                          } else {
                            result[0] += -0.10351018601972084;
                          }
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += -0.06603556442950669;
                    } else {
                      result[0] += -0.00879989389491655;
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)240501391360.0000305) ) ) {
                  result[0] += 0.01746931998780137;
                } else {
                  result[0] += -0.06336030118238382;
                }
              } else {
                result[0] += 0.006916780352180356;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.1822080612182635) ) ) {
          result[0] += 0.41285417140590686;
        } else {
          result[0] += -0.04687277051487031;
        }
      }
    } else {
      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)104.0000000000000142) ) ) {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.154959201812744585) ) ) {
          result[0] += -0.025780088722361396;
        } else {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.743881702423096591) ) ) {
            result[0] += 0.03853294588044516;
          } else {
            result[0] += -0.021832780065702162;
          }
        }
      } else {
        result[0] += -0.02651729910869128;
      }
    }
  } else {
    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
      result[0] += -0.042942040827029536;
    } else {
      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.342454433441162998) ) ) {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
          result[0] += -0.03841352915825462;
        } else {
          result[0] += 0.036329992058765116;
        }
      } else {
        result[0] += 0.019099006647518516;
      }
    }
  }
  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
      if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)11.50000000000000178) ) ) {
        if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
            result[0] += 0.00364368601750504;
          } else {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.445705175399781162) ) ) {
              result[0] += 0.0024963139070185215;
            } else {
              result[0] += -0.016256879577186224;
            }
          }
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.605039834976196733) ) ) {
              result[0] += -0.08037318785162059;
            } else {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += -0.05037363826249355;
                  } else {
                    result[0] += -0.004444057019545134;
                  }
                } else {
                  result[0] += 0.0011447653352174682;
                }
              } else {
                result[0] += -0.004080744102978518;
              }
            }
          } else {
            if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.484580039978028232) ) ) {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)56.00000000000000711) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.284418344497681552) ) ) {
                    result[0] += -0.009137716811065571;
                  } else {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                      result[0] += -0.015024490776084004;
                    } else {
                      result[0] += 0.13881613682518854;
                    }
                  }
                } else {
                  result[0] += 0.004674273505523639;
                }
              } else {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.249904870986938921) ) ) {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += 0.009340578931553916;
                  } else {
                    result[0] += -0.047313445699167046;
                  }
                } else {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.0891790799829024;
                  } else {
                    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                      if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += -0.03336569582816603;
                      } else {
                        if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                          result[0] += 0.08139844971668003;
                        } else {
                          result[0] += -0.10340241857295301;
                        }
                      }
                    } else {
                      result[0] += -0.05004641158446062;
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)240501391360.0000305) ) ) {
                  result[0] += 0.016992523188687044;
                } else {
                  result[0] += -0.06374133992773777;
                }
              } else {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)80.00000000000001421) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.284418344497681552) ) ) {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
                        result[0] += 0.08733383571012537;
                      } else {
                        result[0] += 0.00013291168181954035;
                      }
                    } else {
                      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)56.00000000000000711) ) ) {
                        if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)112.0000000000000142) ) ) {
                          result[0] += 0.014808983958084013;
                        } else {
                          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.924581527709961826) ) ) {
                            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.97438240051269709) ) ) {
                              result[0] += -0.08459620487645751;
                            } else {
                              result[0] += 0.054889680509688114;
                            }
                          } else {
                            result[0] += 0.15760841611919824;
                          }
                        }
                      } else {
                        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                          if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.076962471008301669) ) ) {
                            result[0] += 0.005637162681376438;
                          } else {
                            result[0] += -0.048433943591551974;
                          }
                        } else {
                          result[0] += 0.01683256908365648;
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                      result[0] += -0.07261595720743022;
                    } else {
                      if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                        if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)40.00000000000000711) ) ) {
                          result[0] += -0.07485784920289454;
                        } else {
                          if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                            result[0] += -0.030594111440071348;
                          } else {
                            result[0] += 0.020025296038202564;
                          }
                        }
                      } else {
                        if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                          result[0] += 0.0014713110809649486;
                        } else {
                          result[0] += 0.047466589584449145;
                        }
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.284418344497681552) ) ) {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)88.00000000000001421) ) ) {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.516936540603638583) ) ) {
                        result[0] += 0.024963491175172037;
                      } else {
                        result[0] += -0.1377475070286792;
                      }
                    } else {
                      if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                        result[0] += -0.05680733775170055;
                      } else {
                        result[0] += -0.008876044741867204;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.700598716735840066) ) ) {
                        result[0] += -0.025964861087538895;
                      } else {
                        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2202.000000000000455) ) ) {
                          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)120.0000000000000142) ) ) {
                            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                              result[0] += 0.0636429035940614;
                            } else {
                              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                                result[0] += 0.03152297866788162;
                              } else {
                                result[0] += -0.09829844683781812;
                              }
                            }
                          } else {
                            result[0] += 0.0736578271260978;
                          }
                        } else {
                          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
                              result[0] += -0.09214694942275492;
                            } else {
                              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.339395284652710849) ) ) {
                                result[0] += 0.01461993356983932;
                              } else {
                                result[0] += 0.06147291930257024;
                              }
                            }
                          } else {
                            result[0] += 0.006006872495137744;
                          }
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.970085620880127397) ) ) {
                          result[0] += -0.10675138008404883;
                        } else {
                          result[0] += 0.026108722014612396;
                        }
                      } else {
                        result[0] += -0.15684436958770373;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      } else {
        result[0] += -0.04240202114590425;
      }
    } else {
      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)104.0000000000000142) ) ) {
        if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.778982400894165927) ) ) {
            result[0] += 0.05276718272018861;
          } else {
            result[0] += -0.07268800848592853;
          }
        } else {
          if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
            if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += 0.02324881113362403;
            } else {
              if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.008047734110975157;
              } else {
                result[0] += -0.16895348184970677;
              }
            }
          } else {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
              result[0] += -0.00013935094632837533;
            } else {
              result[0] += 0.051603656607565766;
            }
          }
        }
      } else {
        result[0] += -0.026009505590971056;
      }
    }
  } else {
    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
      result[0] += -0.03949791278194515;
    } else {
      result[0] += 0.014065496096188127;
    }
  }
  if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)7.500000000000000888) ) ) {
      if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
        if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.500000000000000888) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.2687106132507342) ) ) {
            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
              result[0] += 0.008878369732178091;
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.497866153717041238) ) ) {
                result[0] += 0.0024287292019070698;
              } else {
                if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.015417512240113854;
                  } else {
                    result[0] += -0.05165171255269102;
                  }
                } else {
                  result[0] += -0.058466805197952;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += 0.019724838749819222;
            } else {
              result[0] += 0.09752371551053801;
            }
          }
        } else {
          result[0] += -0.08737308181816117;
        }
      } else {
        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.284418344497681552) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.599987030029298651) ) ) {
                result[0] += -0.09396880639308763;
              } else {
                if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.03351415605041871;
                } else {
                  if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.56849193572998225) ) ) {
                      result[0] += -0.04617620947929921;
                    } else {
                      result[0] += 0.138319994556766;
                    }
                  } else {
                    result[0] += -0.058001831166864096;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.861792564392090288) ) ) {
                result[0] += 0.01159816066135066;
              } else {
                result[0] += -0.05899614702821074;
              }
            }
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.802696108818054643) ) ) {
              result[0] += -0.02267875004210156;
            } else {
              if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.023222928512817896;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.599987030029298651) ) ) {
                  result[0] += -0.06538160985441079;
                } else {
                  if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.039720773696899636) ) ) {
                    result[0] += 0.038830419312918495;
                  } else {
                    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)112.0000000000000142) ) ) {
                      result[0] += -0.0368403561639248;
                    } else {
                      result[0] += 0.11040249954259972;
                    }
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
              result[0] += -0.10183278987762817;
            } else {
              result[0] += -0.03302393233007502;
            }
          } else {
            if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.500000000000000444) ) ) {
              if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.376888275146486151) ) ) {
                    result[0] += 0.10949742980650248;
                  } else {
                    result[0] += -0.050067671763723245;
                  }
                } else {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
                    result[0] += -0.09347188762309268;
                  } else {
                    result[0] += 0.019047870061108297;
                  }
                }
              } else {
                result[0] += -0.11090896983940267;
              }
            } else {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)104.0000000000000142) ) ) {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)80.00000000000001421) ) ) {
                  if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)5.000000000000000888) ) ) {
                    if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.076962471008301669) ) ) {
                      result[0] += 0.007423742779822862;
                    } else {
                      result[0] += 0.07442274762619243;
                    }
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.484580039978028232) ) ) {
                      result[0] += 0.003800536359938536;
                    } else {
                      result[0] += -0.05113777735743982;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.249904870986938921) ) ) {
                    if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
                      result[0] += -0.07159753112784814;
                    } else {
                      result[0] += 0.07437308127369818;
                    }
                  } else {
                    if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.918272972106934482) ) ) {
                      result[0] += 0.08806182805660068;
                    } else {
                      result[0] += 0.022983025476726572;
                    }
                  }
                }
              } else {
                result[0] += -0.023485844906802278;
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.970608234405518466) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.597323656082154208) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.481347560882569248) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
                result[0] += -0.023732598198851332;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.357691764831543413) ) ) {
                  result[0] += -0.05368370659599586;
                } else {
                  result[0] += 0.046891522798691694;
                }
              }
            } else {
              result[0] += -0.05652016130219997;
            }
          } else {
            result[0] += -0.048946069463716366;
          }
        } else {
          result[0] += -0.08632571208808675;
        }
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.094205617904663974) ) ) {
          if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.500000000000000444) ) ) {
            result[0] += 0.048845297245968575;
          } else {
            result[0] += -0.009944203147588877;
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
            result[0] += 0.025803337216834723;
          } else {
            result[0] += -0.05843484147457165;
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
      if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.615554332733154741) ) ) {
        result[0] += 0.06367304756444185;
      } else {
        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.487163543701172763) ) ) {
              if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)7.500000000000000888) ) ) {
                result[0] += -0.05571296839785479;
              } else {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)80.00000000000001421) ) ) {
                  if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                    result[0] += -0.046176150512236405;
                  } else {
                    result[0] += 0.021795553170454385;
                  }
                } else {
                  result[0] += -0.049743805919139135;
                }
              }
            } else {
              if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)6.500000000000000888) ) ) {
                result[0] += -0.003529769029036937;
              } else {
                result[0] += 0.20099317169583375;
              }
            }
          } else {
            result[0] += 0.03891402967680813;
          }
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.246492147445679599) ) ) {
            result[0] += -0.08696849842155992;
          } else {
            result[0] += 0.04597546424447126;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.79870843887329279) ) ) {
        result[0] += -0.00021320775943583148;
      } else {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += -0.0810794897477282;
          } else {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
              result[0] += 0.023047059570880473;
            } else {
              result[0] += -0.051540452256238795;
            }
          }
        } else {
          if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += 0.039007065819673226;
          } else {
            result[0] += -0.008216047820399626;
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)7.500000000000000888) ) ) {
      result[0] += 0.0026081543853676867;
    } else {
      result[0] += 0.018921055632627932;
    }
  } else {
    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.172047138214112216) ) ) {
        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.790835380554201) ) ) {
            result[0] += 0.09824158001383698;
          } else {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.284418344497681552) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.780340790748596635) ) ) {
                result[0] += -0.05204679408887404;
              } else {
                result[0] += 0.035482273349642456;
              }
            } else {
              if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                result[0] += -0.028807171692765955;
              } else {
                if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)7.500000000000000888) ) ) {
                  result[0] += 0.01732486429102539;
                } else {
                  if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.329314231872559482) ) ) {
                    result[0] += 0.16309222667771991;
                  } else {
                    result[0] += -0.06231808800231157;
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.040716171264650214) ) ) {
            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += -0.01647664878082867;
            } else {
              result[0] += 0.11157232431064355;
            }
          } else {
            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)1.354025125503540261) ) ) {
              result[0] += -0.06778519416747888;
            } else {
              result[0] += 0.01365155749467071;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.012675821781158891) ) ) {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)88.00000000000001421) ) ) {
            if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)7.000000000000000888) ) ) {
              result[0] += 0.028916481347606944;
            } else {
              result[0] += 0.0762397838510723;
            }
          } else {
            result[0] += -0.020419282151699372;
          }
        } else {
          if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
              if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.607751369476319248) ) ) {
                  result[0] += -0.04823865349951053;
                } else {
                  result[0] += 0.007775810068722397;
                }
              } else {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.90474271774292081) ) ) {
                    result[0] += -0.07095524529965665;
                  } else {
                    result[0] += 0.05649051351590648;
                  }
                } else {
                  if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)5.500000000000000888) ) ) {
                    result[0] += 0.020428977862179447;
                  } else {
                    result[0] += -0.01111672291546359;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)80.00000000000001421) ) ) {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                  result[0] += -0.0353782743890231;
                } else {
                  result[0] += 0.03599461537551218;
                }
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.206118345260621005) ) ) {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.040673396710091375;
                  } else {
                    result[0] += 0.05282077271063445;
                  }
                } else {
                  result[0] += 0.10700979737990952;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.516936540603638583) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                result[0] += -0.09261560734227313;
              } else {
                if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                    result[0] += -0.07928876797765612;
                  } else {
                    result[0] += 0.04241651479114117;
                  }
                } else {
                  result[0] += -0.10575120187469789;
                }
              }
            } else {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                result[0] += -0.05929437369358653;
              } else {
                result[0] += 0.023644793350947185;
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.869339942932130683) ) ) {
          result[0] += -0.09611705789618191;
        } else {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.827801465988160068) ) ) {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.44140100479126021) ) ) {
                result[0] += 0.03635208017704378;
              } else {
                result[0] += -0.16904867188713435;
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.154959201812744585) ) ) {
                result[0] += 0.05099921946704386;
              } else {
                result[0] += -0.041100346014653205;
              }
            }
          } else {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
              result[0] += 0.029378806804038916;
            } else {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.04636259568450371;
                } else {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += 0.06368766090473003;
                  } else {
                    result[0] += -0.1332024779834415;
                  }
                }
              } else {
                result[0] += -0.09377461461289466;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.481347560882569248) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.827801465988160068) ) ) {
            result[0] += 0.015002462013466717;
          } else {
            result[0] += -0.08977565867108263;
          }
        } else {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)72.00000000000001421) ) ) {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2202.000000000000455) ) ) {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.039720773696899636) ) ) {
                if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)15.50000000000000178) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
                      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                          result[0] += -0.017610662493724782;
                        } else {
                          result[0] += 0.02002404597397082;
                        }
                      } else {
                        result[0] += 0.01429611621661352;
                      }
                    } else {
                      result[0] += -0.02529444936637397;
                    }
                  } else {
                    result[0] += 0.0281281988194124;
                  }
                } else {
                  result[0] += -0.031287263232141586;
                }
              } else {
                if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += -0.03810827927566548;
                  } else {
                    result[0] += 0.05484205002318126;
                  }
                } else {
                  result[0] += -0.07053527290157945;
                }
              }
            } else {
              if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)8.500000000000001776) ) ) {
                result[0] += 0.008736058356413669;
              } else {
                result[0] += -0.06693019530415174;
              }
            }
          } else {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.861792564392090288) ) ) {
                result[0] += -0.011078793270981711;
              } else {
                if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.342454433441162998) ) ) {
                    result[0] += 0.08240570251797151;
                  } else {
                    result[0] += -0.05576815764054016;
                  }
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.284418344497681552) ) ) {
                    result[0] += -0.014863417732875573;
                  } else {
                    result[0] += 0.056285288612321065;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += 0.04431583478250918;
              } else {
                result[0] += -0.05097957882038739;
              }
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)20.00000000000000355) ) ) {
    if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
      if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)4.500000000000000888) ) ) {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.154959201812744585) ) ) {
          if ( LIKELY( !(data[51].missing != -1) || (data[51].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += 0.003558499444588775;
          } else {
            result[0] += 0.055184715752041105;
          }
        } else {
          if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)11.50000000000000178) ) ) {
            result[0] += 0.002976212776168219;
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.651049375534058505) ) ) {
              if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.142630577087403232) ) ) {
                  result[0] += 0.03724037930044103;
                } else {
                  result[0] += -0.0519260058077802;
                }
              } else {
                result[0] += 0.04383500278750803;
              }
            } else {
              result[0] += -0.05418702435951395;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)10.50000000000000178) ) ) {
          if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)9.500000000000001776) ) ) {
            if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
              result[0] += -0.10745900016955767;
            } else {
              result[0] += -0.002079906588363763;
            }
          } else {
            if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)15.50000000000000178) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                result[0] += -0.0325240663309495;
              } else {
                result[0] += 0.03458063255428768;
              }
            } else {
              if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.851041555404663974) ) ) {
                  result[0] += -0.00037182409761859743;
                } else {
                  if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)9.500000000000001776) ) ) {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.172047138214112216) ) ) {
                      result[0] += -0.08672844203727864;
                    } else {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                        result[0] += 0.35256408220511937;
                      } else {
                        result[0] += 0.9365734600592897;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.909855604171753818) ) ) {
                        if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)22.50000000000000355) ) ) {
                          result[0] += -0.10004038324383467;
                        } else {
                          result[0] += 0.046118348868660085;
                        }
                      } else {
                        result[0] += -0.08601670897555619;
                      }
                    } else {
                      result[0] += -0.09240139541974948;
                    }
                  }
                }
              } else {
                result[0] += -0.07144098545821623;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.350240230560303178) ) ) {
              result[0] += -0.02654975864218075;
            } else {
              if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)14.50000000000000178) ) ) {
                  result[0] += 0.050865768641926185;
                } else {
                  result[0] += -0.10723955210613721;
                }
              } else {
                result[0] += -0.06216705351402233;
              }
            }
          } else {
            if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += 0.033442723060716995;
              } else {
                result[0] += -0.003587430531824592;
              }
            } else {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                result[0] += 0.007653028388613899;
              } else {
                result[0] += -0.056219046172638114;
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.802696108818054643) ) ) {
        result[0] += -0.031945470081641426;
      } else {
        if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
          result[0] += 0.022204231541229594;
        } else {
          if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
            result[0] += -0.03221134746864861;
          } else {
            if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)16.50000000000000355) ) ) {
              result[0] += 0.028510693340510075;
            } else {
              result[0] += -0.1026807410631659;
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.736135363578796831) ) ) {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.671254158020020419) ) ) {
              result[0] += -0.020394947901544;
            } else {
              if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)1.354025125503540261) ) ) {
                  result[0] += 0.03331386882424842;
                } else {
                  result[0] += -0.04525040635899962;
                }
              } else {
                result[0] += -0.1308281137819801;
              }
            }
          } else {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.134366035461426669) ) ) {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.802901029586792436) ) ) {
                    result[0] += 0.13537720093281094;
                  } else {
                    if ( UNLIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += -0.0799679134821519;
                    } else {
                      result[0] += 0.08081814417244787;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.802901029586792436) ) ) {
                    result[0] += -0.1004364121713503;
                  } else {
                    result[0] += 0.06049912735691862;
                  }
                }
              } else {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.03960975050400328;
                } else {
                  result[0] += 0.005123659406748464;
                }
              }
            } else {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += -0.07133219256661388;
              } else {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                  result[0] += -0.0811750750218263;
                } else {
                  if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.05143093334600381;
                  } else {
                    result[0] += -0.019331191310029545;
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.34969997406006037) ) ) {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
              result[0] += 0.17321542809249757;
            } else {
              result[0] += -0.07081326881903913;
            }
          } else {
            result[0] += 0.016208578789358618;
          }
        }
      } else {
        if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
          result[0] += -0.049786232778874256;
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.41211462020874201) ) ) {
            result[0] += -0.05701342299969503;
          } else {
            result[0] += 0.0010852232607714725;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)7.000000000000000888) ) ) {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.42478513717651456) ) ) {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.849175214767456943) ) ) {
            result[0] += 0.003063841028429571;
          } else {
            result[0] += -0.02004810992258148;
          }
        } else {
          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
            if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
              result[0] += 0.06409858432718367;
            } else {
              result[0] += -0.019892539987359806;
            }
          } else {
            result[0] += -0.052668933695002466;
          }
        }
      } else {
        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.715336322784424716) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
            result[0] += -0.06039685582751825;
          } else {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.778982400894165927) ) ) {
              result[0] += -0.03350681269756925;
            } else {
              result[0] += 0.011811331243213105;
            }
          }
        } else {
          result[0] += 0.05638014254648277;
        }
      }
    }
  }
  if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)4.500000000000000888) ) ) {
      if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
        if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
          result[0] += 0.0008032456534777009;
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.357691764831543413) ) ) {
            result[0] += -0.0697127360359134;
          } else {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.357691764831543413) ) ) {
              result[0] += 0.14635372798244223;
            } else {
              if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += -0.07691698262343294;
              } else {
                if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.04924427801022817;
                } else {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)120.0000000000000142) ) ) {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                      result[0] += -0.008225039746270004;
                    } else {
                      result[0] += 0.03842684765024018;
                    }
                  } else {
                    result[0] += -0.017602924180915178;
                  }
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.802696108818054643) ) ) {
          result[0] += -0.03075375047632613;
        } else {
          if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)2.500000000000000444) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.029068946838379794) ) ) {
              result[0] += 0.005545760657904738;
            } else {
              result[0] += 0.044193459923193;
            }
          } else {
            result[0] += 0.006194018804881797;
          }
        }
      }
    } else {
      result[0] += 0.09407805115973633;
    }
  } else {
    if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)56.00000000000000711) ) ) {
      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.802696108818054643) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.184516429901124823) ) ) {
          result[0] += 0.04584119074927346;
        } else {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
            result[0] += -0.015431557254349144;
          } else {
            result[0] += 0.007772898448794104;
          }
        }
      } else {
        if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)4.500000000000000888) ) ) {
          result[0] += -0.038653962434336164;
        } else {
          result[0] += -0.002271800997099176;
        }
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.02604460716247603) ) ) {
        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
          result[0] += -0.018329338694215103;
        } else {
          result[0] += 0.00849456623428073;
        }
      } else {
        if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.357691764831543413) ) ) {
                result[0] += -0.17472871371522236;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.827801465988160068) ) ) {
                  result[0] += 0.027359730033164043;
                } else {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                    result[0] += -0.01206272091898691;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.040716171264650214) ) ) {
                      result[0] += 0.10286842443376482;
                    } else {
                      result[0] += -0.08079800344946349;
                    }
                  }
                }
              }
            } else {
              result[0] += 0.06995041151625314;
            }
          } else {
            result[0] += -0.05827164923193954;
          }
        } else {
          if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
            if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.500000000000000444) ) ) {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += -0.0417661678941125;
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.131699204444885698) ) ) {
                  result[0] += -0.07994574374855509;
                } else {
                  result[0] += 0.024686549995400944;
                }
              }
            } else {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.599987030029298651) ) ) {
                      if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                        result[0] += 0.10847562796681792;
                      } else {
                        result[0] += -0.03067674940468165;
                      }
                    } else {
                      if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                        if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.44140100479126021) ) ) {
                          result[0] += -0.05576067339051462;
                        } else {
                          result[0] += 0.12204456050425459;
                        }
                      } else {
                        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)80.00000000000001421) ) ) {
                          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.088880300521851474) ) ) {
                            result[0] += 0.028942589092111105;
                          } else {
                            result[0] += -0.05681548815228957;
                          }
                        } else {
                          result[0] += -0.04787102568440366;
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += 0.002634266022541306;
                    } else {
                      result[0] += 0.087520424655372;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.842307567596437323) ) ) {
                      result[0] += 0.09155999745507692;
                    } else {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.172047138214112216) ) ) {
                        result[0] += -0.06304521083443107;
                      } else {
                        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.379217386245728427) ) ) {
                            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
                              if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)3.500000000000000444) ) ) {
                                result[0] += 0.03683678029679993;
                              } else {
                                result[0] += -0.09564934345988657;
                              }
                            } else {
                              result[0] += 0.028771395255579826;
                            }
                          } else {
                            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                              result[0] += -0.0174517778542709;
                            } else {
                              result[0] += 0.11238422935328443;
                            }
                          }
                        } else {
                          result[0] += 0.12298067009721066;
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.834949493408204901) ) ) {
                      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                        result[0] += -0.042314085519087;
                      } else {
                        result[0] += -0.13816256146240483;
                      }
                    } else {
                      if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.44140100479126021) ) ) {
                        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.339395284652710849) ) ) {
                            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)112.0000000000000142) ) ) {
                              result[0] += -0.01564187030341171;
                            } else {
                              if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                                result[0] += 0.045977220324983414;
                              } else {
                                result[0] += 0.007850395721794298;
                              }
                            }
                          } else {
                            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                              result[0] += 0.02588358528132581;
                            } else {
                              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                                result[0] += -0.013845103977023418;
                              } else {
                                result[0] += -0.09121732808045893;
                              }
                            }
                          }
                        } else {
                          result[0] += -0.037000965426736;
                        }
                      } else {
                        result[0] += -0.16108098257267567;
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.97438240051269709) ) ) {
                  if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.039720773696899636) ) ) {
                    result[0] += -0.11475169745891985;
                  } else {
                    result[0] += -0.04432537563515214;
                  }
                } else {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += 0.015328560271408602;
                  } else {
                    result[0] += -0.07608746361501527;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.45958471298217951) ) ) {
              result[0] += -0.03642832209957803;
            } else {
              result[0] += 0.039950702549321856;
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.1822080612182635) ) ) {
        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)80.00000000000001421) ) ) {
          if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
            result[0] += -0.0003844135688315832;
          } else {
            result[0] += -0.05704485669931837;
          }
        } else {
          if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.039720773696899636) ) ) {
            result[0] += -0.018134411870360547;
          } else {
            result[0] += 0.0386104357007936;
          }
        }
      } else {
        if ( LIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.172047138214112216) ) ) {
            result[0] += 0.016856753615507505;
          } else {
            if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)10.50000000000000178) ) ) {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += 0.01581326864086967;
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.651049375534058505) ) ) {
                  result[0] += 0.014094675925521183;
                } else {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                    result[0] += -0.03763693408963686;
                  } else {
                    result[0] += -0.13001262039326963;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)29.50000000000000355) ) ) {
                result[0] += -0.054105583287555895;
              } else {
                result[0] += 0.013567285263413257;
              }
            }
          }
        } else {
          result[0] += -0.05055361545284026;
        }
      }
    } else {
      if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
            result[0] += -0.1958899733210591;
          } else {
            result[0] += 0.1070318816974561;
          }
        } else {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)104.0000000000000142) ) ) {
            result[0] += -0.005536550432253269;
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.909855604171753818) ) ) {
              result[0] += -0.0869119128295916;
            } else {
              result[0] += 0.026881750658548576;
            }
          }
        }
      } else {
        result[0] += 0.0128540020584397;
      }
    }
  } else {
    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.172047138214112216) ) ) {
        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
          if ( LIKELY( !(data[51].missing != -1) || (data[51].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.481347560882569248) ) ) {
                result[0] += 0.7832070079344522;
              } else {
                result[0] += -0.051509133404351405;
              }
            } else {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.357691764831543413) ) ) {
                  result[0] += -0.043048609923273765;
                } else {
                  result[0] += 0.002221448494480449;
                }
              } else {
                result[0] += 0.045456725895542595;
              }
            }
          } else {
            result[0] += 0.07089069841962402;
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.38936424255371271) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.669892311096192294) ) ) {
              result[0] += 0.027479861787806932;
            } else {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)1.354025125503540261) ) ) {
                if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.18292076364099907;
                } else {
                  if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.07669892943185111;
                  } else {
                    if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += -0.05262480180243191;
                      } else {
                        result[0] += 0.004158031023598811;
                      }
                    } else {
                      result[0] += 0.03347200338882005;
                    }
                  }
                }
              } else {
                result[0] += 0.10909815198007367;
              }
            }
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.780340790748596635) ) ) {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.11111537850098877;
                } else {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.743881702423096591) ) ) {
                      result[0] += -0.07709417830827452;
                    } else {
                      result[0] += -0.02187147400223418;
                    }
                  } else {
                    result[0] += 0.073017962162866;
                  }
                }
              } else {
                if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.595119953155518466) ) ) {
                  result[0] += -0.030653717060175136;
                } else {
                  result[0] += -0.08501708810069886;
                }
              }
            } else {
              result[0] += 0.017698551369920615;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.516936540603638583) ) ) {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
            result[0] += 0.008116092251436323;
          } else {
            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.0650703355508511;
              } else {
                result[0] += -0.012750831709387898;
              }
            } else {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += 0.015008051538593635;
              } else {
                result[0] += 0.20134534187507094;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)2.500000000000000444) ) ) {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.058282586152721486;
            } else {
              result[0] += 0.008640017969279903;
            }
          } else {
            if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)6.500000000000000888) ) ) {
              result[0] += -0.07368573085646676;
            } else {
              if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.02228928982085711;
              } else {
                if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)27.50000000000000355) ) ) {
                  result[0] += -0.06573097355440059;
                } else {
                  if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2565.000000000000455) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.070054531097412998) ) ) {
                      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                        result[0] += 0.04319378449938964;
                      } else {
                        result[0] += -0.04413763043637335;
                      }
                    } else {
                      result[0] += -0.08707899993661361;
                    }
                  } else {
                    result[0] += 0.7853303894496797;
                  }
                }
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.516936540603638583) ) ) {
        if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
          result[0] += 0.0034460092905703277;
        } else {
          result[0] += 0.021130421815803612;
        }
      } else {
        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.44140100479126021) ) ) {
          result[0] += -0.0780907668747801;
        } else {
          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
              if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)15.50000000000000178) ) ) {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                  result[0] += -0.010036562023524924;
                } else {
                  result[0] += -0.053950796683149865;
                }
              } else {
                result[0] += 0.018805899514344415;
              }
            } else {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.607751369476319248) ) ) {
                result[0] += 0.04121879980475516;
              } else {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.417592287063599077) ) ) {
                  result[0] += -0.0697972888460651;
                } else {
                  result[0] += 0.013288194932455805;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
              result[0] += 0.0175576463525292;
            } else {
              result[0] += -0.08611615364559477;
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)3.000000000000000444) ) ) {
    if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
        result[0] += 0.08892493557675006;
      } else {
        result[0] += 0.007305117934394876;
      }
    } else {
      if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)6.500000000000000888) ) ) {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.142630577087403232) ) ) {
          if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)2.500000000000000444) ) ) {
            if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)15.50000000000000178) ) ) {
              result[0] += -0.12711800875713852;
            } else {
              result[0] += 0.023645389336287015;
            }
          } else {
            if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)4.500000000000000888) ) ) {
              result[0] += 0.10460518300879747;
            } else {
              result[0] += -0.010733700933944514;
            }
          }
        } else {
          if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)2.500000000000000444) ) ) {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.124530076980591708) ) ) {
                  result[0] += -0.02031427054834262;
                } else {
                  result[0] += 0.1478206405900707;
                }
              } else {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                  result[0] += 0.0431493667619001;
                } else {
                  result[0] += -0.07267919717012157;
                }
              }
            } else {
              result[0] += -0.04465553605574671;
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
              result[0] += -0.005674216480917263;
            } else {
              result[0] += -0.05476497656130666;
            }
          }
        }
      } else {
        result[0] += 0.0014928060932921298;
      }
    }
  } else {
    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.602003335952759233) ) ) {
        if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)7.500000000000000888) ) ) {
          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
            if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.129137588461695;
            } else {
              result[0] += -0.0899945082615663;
            }
          } else {
            result[0] += -0.004960040688650908;
          }
        } else {
          result[0] += 0.03709912917682325;
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.651049375534058505) ) ) {
          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)20.00000000000000355) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.10290670394897639) ) ) {
              result[0] += -0.028908008248479368;
            } else {
              result[0] += 0.03539361380553449;
            }
          } else {
            result[0] += -0.0151308059832377;
          }
        } else {
          if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += -0.035600363252018395;
          } else {
            result[0] += -0.081722552290481;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
        if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)240501391360.0000305) ) ) {
          result[0] += -0.00018092018482090006;
        } else {
          result[0] += -0.08948567293463622;
        }
      } else {
        if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)6.000000000000000888) ) ) {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
            if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
              result[0] += 0.33830811645191794;
            } else {
              result[0] += 0.022207843725284893;
            }
          } else {
            result[0] += -0.04263235702663423;
          }
        } else {
          if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.605039834976196733) ) ) {
                result[0] += -0.0812490708971374;
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.605039834976196733) ) ) {
                  result[0] += -0.04950502650635978;
                } else {
                  result[0] += -0.004326329293319032;
                }
              }
            } else {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)80.00000000000001421) ) ) {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.484580039978028232) ) ) {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.172047138214112216) ) ) {
                      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.040716171264650214) ) ) {
                          result[0] += -0.05628161189807497;
                        } else {
                          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                            result[0] += 0.023730746559211157;
                          } else {
                            result[0] += -0.008519408243486203;
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.04969787597656428) ) ) {
                          result[0] += 0.006519140912835696;
                        } else {
                          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.585059762001038042) ) ) {
                            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                                result[0] += -0.06140851561985722;
                              } else {
                                result[0] += -0.015324003178572326;
                              }
                            } else {
                              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                                result[0] += -0.029599934442673753;
                              } else {
                                result[0] += 0.07984066629346454;
                              }
                            }
                          } else {
                            result[0] += 0.07477688340108277;
                          }
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += -0.01202395442674651;
                      } else {
                        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)120.0000000000000142) ) ) {
                          result[0] += 0.04377385212854542;
                        } else {
                          result[0] += 0.13731039882535379;
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.869339942932130683) ) ) {
                      if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += 0.002063709393623484;
                      } else {
                        result[0] += -0.058409227933699764;
                      }
                    } else {
                      if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                        result[0] += -0.04081544825278535;
                      } else {
                        result[0] += 0.036519787475363524;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.475347042083742011) ) ) {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                      if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                        result[0] += -0.10934924217667466;
                      } else {
                        result[0] += 0.028000613699035927;
                      }
                    } else {
                      result[0] += -0.037527246575159934;
                    }
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.339395284652710849) ) ) {
                      if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += -0.032065057111539334;
                      } else {
                        if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                          if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                            result[0] += -0.08481045864601268;
                          } else {
                            result[0] += 0.056268195938005486;
                          }
                        } else {
                          result[0] += -0.013547161476122;
                        }
                      }
                    } else {
                      result[0] += -0.03502882464608465;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.484580039978028232) ) ) {
                  result[0] += -0.046552922512052376;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.834949493408204901) ) ) {
                    result[0] += -0.03261430324614211;
                  } else {
                    result[0] += 0.0261074304352209;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)104.0000000000000142) ) ) {
              if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.029605211155282685;
                } else {
                  if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)7.500000000000000888) ) ) {
                    result[0] += -0.16907483888792038;
                  } else {
                    result[0] += 0.016978836924593895;
                  }
                }
              } else {
                result[0] += 0.04794267730860149;
              }
            } else {
              result[0] += -0.019978802563259207;
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY(  (data[27].missing != -1) && (data[27].fvalue <= (double)-1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)240501391360.0000305) ) ) {
      if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.02604460716247603) ) ) {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += -0.5780592771238924;
            } else {
              result[0] += 0.010560077559002066;
            }
          } else {
            result[0] += 0.10652420172626098;
          }
        } else {
          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += 0.029751355683476616;
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.357691764831543413) ) ) {
              result[0] += -0.018681064486311814;
            } else {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.141444921493531162) ) ) {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.607751369476319248) ) ) {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.851041555404663974) ) ) {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.255632162094117099) ) ) {
                        result[0] += -0.037763416341506165;
                      } else {
                        result[0] += -0.14423486484462633;
                      }
                    } else {
                      result[0] += 0.0615773426711337;
                    }
                  } else {
                    result[0] += -0.12082672423810906;
                  }
                } else {
                  result[0] += 0.09671338826388196;
                }
              } else {
                result[0] += -0.13286485001007503;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
          if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)6.117118597030640537) ) ) {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.602003335952759233) ) ) {
              result[0] += 0.02436471039516751;
            } else {
              result[0] += -3.7338468109291415e-05;
            }
          } else {
            result[0] += 0.04057067192851336;
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.497866153717041238) ) ) {
            result[0] += 0.02303045222761564;
          } else {
            result[0] += 0.09090010012231903;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.497866153717041238) ) ) {
        result[0] += 0.01809247482398369;
      } else {
        result[0] += 0.06619411039192889;
      }
    }
  } else {
    if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)6.500000000000000888) ) ) {
      if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
        result[0] += -0.001968960466453079;
      } else {
        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)168.0000000000000284) ) ) {
          result[0] += -0.01164372088239106;
        } else {
          result[0] += -0.09387119134633211;
        }
      }
    } else {
      if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)9.500000000000001776) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
          result[0] += 0.06895963803431461;
        } else {
          if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)16.50000000000000355) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
              result[0] += 0.06959279983696497;
            } else {
              if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.0718177645854015;
              } else {
                result[0] += 0.03211093452427351;
              }
            }
          } else {
            result[0] += -0.02900296183721837;
          }
        }
      } else {
        if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.154959201812744585) ) ) {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.0018956564140525255;
            } else {
              result[0] += -0.08400089282997852;
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.357691764831543413) ) ) {
              result[0] += -0.058598157051285876;
            } else {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.597323656082154208) ) ) {
                  if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.970608234405518466) ) ) {
                    result[0] += 0.03737544679020418;
                  } else {
                    result[0] += -0.07679093164111948;
                  }
                } else {
                  result[0] += -0.04748621134866233;
                }
              } else {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.743881702423096591) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.255632162094117099) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.172047138214112216) ) ) {
                        result[0] += -0.0951985545100135;
                      } else {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
                          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                            result[0] += -0.11113795948317669;
                          } else {
                            if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                              result[0] += 0.029439985188635588;
                            } else {
                              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.993164777755738193) ) ) {
                                result[0] += -0.1590960084031292;
                              } else {
                                result[0] += -0.006291642726647348;
                              }
                            }
                          }
                        } else {
                          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                            result[0] += -0.08582623651641155;
                          } else {
                            result[0] += 0.02520865744119671;
                          }
                        }
                      }
                    } else {
                      result[0] += 0.016011954929715834;
                    }
                  } else {
                    result[0] += 0.02138722592441566;
                  }
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.255632162094117099) ) ) {
                    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.484580039978028232) ) ) {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.467917680740357333) ) ) {
                          result[0] += -0.093730200328116;
                        } else {
                          result[0] += 0.01622213619061916;
                        }
                      } else {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
                          result[0] += -0.09430057260646257;
                        } else {
                          result[0] += 0.03151839061140802;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.141444921493531162) ) ) {
                        if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.970608234405518466) ) ) {
                          result[0] += 0.07356801777436302;
                        } else {
                          result[0] += -0.04541717119991016;
                        }
                      } else {
                        result[0] += 0.004393333368387957;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                      result[0] += 0.029097870748565166;
                    } else {
                      result[0] += -0.020166426114695258;
                    }
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.736135363578796831) ) ) {
              result[0] += 0.056634286361728105;
            } else {
              result[0] += -0.07440825163200525;
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.094205617904663974) ) ) {
              if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)5.500000000000000888) ) ) {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += 0.035046838752485085;
                } else {
                  result[0] += 0.005745659799480715;
                }
              } else {
                result[0] += -0.04343592119342497;
              }
            } else {
              if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.357691764831543413) ) ) {
                  result[0] += 0.008217952225472868;
                } else {
                  result[0] += -0.06548121616080499;
                }
              } else {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
                    result[0] += 0.013804024618060642;
                  } else {
                    result[0] += -0.04875631437428002;
                  }
                } else {
                  if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)240501391360.0000305) ) ) {
                    if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)16.50000000000000355) ) ) {
                      result[0] += 0.007516791881676027;
                    } else {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.617236852645874912) ) ) {
                        result[0] += 0.05267052947334477;
                      } else {
                        result[0] += 0.3127121026393651;
                      }
                    }
                  } else {
                    result[0] += -0.0271539589460959;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY(  (data[46].missing != -1) && (data[46].fvalue <= (double)-1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
        if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
          result[0] += 0.10595424172524204;
        } else {
          result[0] += -0.2642938837415719;
        }
      } else {
        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
          result[0] += 0.03374931403828004;
        } else {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
            result[0] += 0.007143514176545688;
          } else {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.016888196429077747;
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.255632162094117099) ) ) {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                  if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.04014473091492991;
                  } else {
                    result[0] += -0.15081400306682927;
                  }
                } else {
                  result[0] += -0.016717398311572297;
                }
              } else {
                result[0] += 0.08224766197389649;
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.497866153717041238) ) ) {
        result[0] += 0.019829855983213757;
      } else {
        result[0] += 0.06403429958040532;
      }
    }
  } else {
    if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
        if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)11.50000000000000178) ) ) {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.993164777755738193) ) ) {
            if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)5.500000000000000888) ) ) {
                if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.005040830010481704;
                    } else {
                      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                        result[0] += -0.08075274255196557;
                      } else {
                        result[0] += -0.018347911625245687;
                      }
                    }
                  } else {
                    result[0] += -0.04681086724236277;
                  }
                } else {
                  if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)7.000000000000000888) ) ) {
                    result[0] += 0.00012739175944684807;
                  } else {
                    result[0] += -0.023686360586758673;
                  }
                }
              } else {
                result[0] += -0.08197737129995236;
              }
            } else {
              if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)14.50000000000000178) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.361115694046021396) ) ) {
                  if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)12.50000000000000178) ) ) {
                    if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)9.500000000000001776) ) ) {
                      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                        result[0] += 0.06807935251593054;
                      } else {
                        result[0] += 0.030442156779775062;
                      }
                    } else {
                      if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)3.500000000000000444) ) ) {
                        result[0] += 0.07334757968532488;
                      } else {
                        result[0] += 0.006679077769344889;
                      }
                    }
                  } else {
                    result[0] += -0.019027432910449366;
                  }
                } else {
                  if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)10.50000000000000178) ) ) {
                    result[0] += -0.03563284977999522;
                  } else {
                    result[0] += 0.024010592451138818;
                  }
                }
              } else {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.141444921493531162) ) ) {
                    result[0] += -0.16115236303602098;
                  } else {
                    result[0] += 0.03363656815521529;
                  }
                } else {
                  result[0] += 0.0379846704248987;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)72.00000000000001421) ) ) {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)14.50000000000000178) ) ) {
                  if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)11.50000000000000178) ) ) {
                    if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
                      if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                        if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)4.500000000000000888) ) ) {
                          if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                            result[0] += -0.04199800588993116;
                          } else {
                            result[0] += 0.03996661311961493;
                          }
                        } else {
                          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                            result[0] += 0.11301171968349444;
                          } else {
                            result[0] += 0.011590855722736305;
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)6.500000000000000888) ) ) {
                          result[0] += -0.059223690057210565;
                        } else {
                          if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)7.500000000000000888) ) ) {
                            result[0] += 0.04103881594013935;
                          } else {
                            if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)9.500000000000001776) ) ) {
                              result[0] += -0.01630998108183973;
                            } else {
                              result[0] += 0.02372647906816303;
                            }
                          }
                        }
                      }
                    } else {
                      result[0] += -0.04302115870005499;
                    }
                  } else {
                    if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)25.50000000000000355) ) ) {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.651049375534058505) ) ) {
                        result[0] += 0.010170906763401022;
                      } else {
                        result[0] += -0.072514393013751;
                      }
                    } else {
                      if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2565.000000000000455) ) ) {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.909855604171753818) ) ) {
                          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.825422286987305576) ) ) {
                            result[0] += 0.03589555131373972;
                          } else {
                            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                                result[0] += -0.03723644394207884;
                              } else {
                                result[0] += 0.05833746911957338;
                              }
                            } else {
                              result[0] += -0.06829506419800768;
                            }
                          }
                        } else {
                          result[0] += -0.07743143757175164;
                        }
                      } else {
                        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.000308036804200107) ) ) {
                          result[0] += 0.03144125512068091;
                        } else {
                          result[0] += 0.63472060281056;
                        }
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                    result[0] += 0.06292992389969235;
                  } else {
                    result[0] += -0.05277179369240981;
                  }
                }
              } else {
                if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += -0.0718568074662717;
                } else {
                  result[0] += -0.03387481784448834;
                }
              }
            } else {
              if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.07556688645452665;
              } else {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += 0.03725297642913524;
                } else {
                  result[0] += -0.03147449393405959;
                }
              }
            }
          }
        } else {
          result[0] += -0.03784808241852189;
        }
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.154959201812744585) ) ) {
          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += 0.0021837418646937358;
          } else {
            result[0] += -0.09067493843717667;
          }
        } else {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)104.0000000000000142) ) ) {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.42478513717651456) ) ) {
              result[0] += 0.03360640631456979;
            } else {
              result[0] += -0.012652590657294405;
            }
          } else {
            result[0] += -0.021261948746941695;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.284418344497681552) ) ) {
        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.467917680740357333) ) ) {
            result[0] += -0.10052052940395846;
          } else {
            result[0] += -0.013609484219964538;
          }
        } else {
          result[0] += 0.010761778311741124;
        }
      } else {
        result[0] += 0.014618349516053104;
      }
    }
  }
  if ( UNLIKELY(  (data[27].missing != -1) && (data[27].fvalue <= (double)-1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)240501391360.0000305) ) ) {
      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
          result[0] += 0.10552272999981566;
        } else {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
            result[0] += -0.433969593419357;
          } else {
            result[0] += -0.05425133951272514;
          }
        }
      } else {
        if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += 0.024235412087095;
          } else {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.051912069320679599) ) ) {
              if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.05837903198585015;
                } else {
                  result[0] += -0.16522167598171172;
                }
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.339395284652710849) ) ) {
                  result[0] += -0.020824519731074916;
                } else {
                  result[0] += 0.04872186063064339;
                }
              }
            } else {
              result[0] += 0.033398816057716295;
            }
          }
        } else {
          if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)6.117118597030640537) ) ) {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.602003335952759233) ) ) {
              result[0] += 0.023020736829668986;
            } else {
              if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.497866153717041238) ) ) {
                  result[0] += 0.02013344360649157;
                } else {
                  result[0] += 0.09605558126774438;
                }
              } else {
                result[0] += -0.0004121896083398018;
              }
            }
          } else {
            result[0] += 0.0441428715625049;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.497866153717041238) ) ) {
        result[0] += 0.01723632726127789;
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.19876670837402521) ) ) {
          result[0] += 0.015448933109301753;
        } else {
          result[0] += 0.07737087331488568;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)6.500000000000000888) ) ) {
      if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)72.00000000000001421) ) ) {
            result[0] += -0.002993242821054089;
          } else {
            result[0] += -0.01841692596874815;
          }
        } else {
          result[0] += -0.00023808459989998763;
        }
      } else {
        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)168.0000000000000284) ) ) {
          result[0] += -0.011013670438642558;
        } else {
          result[0] += -0.09107688018135601;
        }
      }
    } else {
      if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)9.500000000000001776) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
          result[0] += 0.06154299128038015;
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.154959201812744585) ) ) {
            result[0] += 0.06232926307767649;
          } else {
            if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.56849193572998225) ) ) {
                result[0] += -0.08865470121092571;
              } else {
                result[0] += 0.06235317903305178;
              }
            } else {
              result[0] += 0.024197699861725475;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
          if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.651049375534058505) ) ) {
                if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.736135363578796831) ) ) {
                    result[0] += 0.07983005737665005;
                  } else {
                    result[0] += -0.025322608205508003;
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.154959201812744585) ) ) {
                    result[0] += -0.04632137300951883;
                  } else {
                    result[0] += 0.061354252751520094;
                  }
                }
              } else {
                result[0] += -0.05294857775672384;
              }
            } else {
              result[0] += -0.08837866374237617;
            }
          } else {
            if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)3.500000000000000444) ) ) {
              if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                result[0] += 0.1558376803904025;
              } else {
                result[0] += 0.039423772223028446;
              }
            } else {
              if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)4.500000000000000888) ) ) {
                if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                    result[0] += -0.06931968901108336;
                  } else {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.802696108818054643) ) ) {
                      result[0] += 0.10803345577867592;
                    } else {
                      result[0] += -0.029000843591084725;
                    }
                  }
                } else {
                  result[0] += -0.0916405291735771;
                }
              } else {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.597323656082154208) ) ) {
                    result[0] += 0.016114489865765736;
                  } else {
                    result[0] += -0.04522783889203624;
                  }
                } else {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += -0.06037928737498443;
                  } else {
                    if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)137422176256.0000153) ) ) {
                      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.484580039978028232) ) ) {
                          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.94957673549652144) ) ) {
                            result[0] += -0.024224226374273433;
                          } else {
                            result[0] += 0.11099466698544236;
                          }
                        } else {
                          result[0] += 0.007629380382591342;
                        }
                      } else {
                        result[0] += 0.0073498149003428164;
                      }
                    } else {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.651049375534058505) ) ) {
                        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.56849193572998225) ) ) {
                            if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)9.500000000000001776) ) ) {
                              result[0] += -0.0123607730161649;
                            } else {
                              result[0] += -0.053684689108800435;
                            }
                          } else {
                            result[0] += -0.10078166601203642;
                          }
                        } else {
                          if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.039720773696899636) ) ) {
                            if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)12.50000000000000178) ) ) {
                              result[0] += 0.015993896627788853;
                            } else {
                              result[0] += -0.08083508293499272;
                            }
                          } else {
                            result[0] += -0.07847242674863612;
                          }
                        }
                      } else {
                        if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)5.500000000000000888) ) ) {
                          result[0] += -0.04471828989590049;
                        } else {
                          result[0] += 0.011035120767872964;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.094205617904663974) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.94957673549652144) ) ) {
              result[0] += -0.009931681809040579;
            } else {
              result[0] += -0.21281741782654137;
            }
          } else {
            if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += -0.08680980487863202;
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.357691764831543413) ) ) {
                result[0] += -0.0655150518220382;
              } else {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.827801465988160068) ) ) {
                    result[0] += 0.08622809330885552;
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.342454433441162998) ) ) {
                      result[0] += 0.07219470063091385;
                    } else {
                      result[0] += -0.06071543120134881;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)104.0000000000000142) ) ) {
                    result[0] += 0.055461411982633047;
                  } else {
                    result[0] += -0.00597861413263333;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  if ( LIKELY(  (data[42].missing != -1) && (data[42].fvalue <= (double)-1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.172047138214112216) ) ) {
        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)6.117118597030640537) ) ) {
          result[0] += 0.0029828124265866614;
        } else {
          result[0] += 0.02654459733849783;
        }
      } else {
        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
          result[0] += 0.1095645177219003;
        } else {
          result[0] += 0.014770482057997867;
        }
      }
    } else {
      result[0] += -0.0007791527244377733;
    }
  } else {
    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.602003335952759233) ) ) {
        if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)7.500000000000000888) ) ) {
          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
            if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.1118711343616796;
            } else {
              result[0] += -0.08186402628544005;
            }
          } else {
            if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)8816427008.000001907) ) ) {
              result[0] += 0.2448418803278679;
            } else {
              if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.02269338452578174;
              } else {
                result[0] += 0.01328728773529225;
              }
            }
          }
        } else {
          result[0] += 0.033241443628677474;
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.778982400894165927) ) ) {
          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)20.00000000000000355) ) ) {
            if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.10290670394897639) ) ) {
                result[0] += -0.03938645374353334;
              } else {
                result[0] += 0.025170287279368733;
              }
            } else {
              result[0] += 0.10747149472826643;
            }
          } else {
            result[0] += -0.014195861980543751;
          }
        } else {
          if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.450390577316285068) ) ) {
              result[0] += -0.029213691425162752;
            } else {
              result[0] += -0.08135395826652947;
            }
          } else {
            result[0] += -0.07799609526957661;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
        if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)240501391360.0000305) ) ) {
          result[0] += -0.0023841239100143526;
        } else {
          result[0] += -0.08685797355177952;
        }
      } else {
        if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)6.000000000000000888) ) ) {
          if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.764309883117677558) ) ) {
                result[0] += -0.027388411156468995;
              } else {
                result[0] += 0.1097297114819939;
              }
            } else {
              result[0] += 0.36661767086146213;
            }
          } else {
            result[0] += -0.04593446052096667;
          }
        } else {
          if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.605039834976196733) ) ) {
                result[0] += -0.0761141113660102;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.1822080612182635) ) ) {
                  if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)7.000000000000000888) ) ) {
                    result[0] += 0.0016323624410826646;
                  } else {
                    result[0] += -0.06024185515517222;
                  }
                } else {
                  if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                    result[0] += 0.00764775596349188;
                  } else {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)80.00000000000001421) ) ) {
                        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                          result[0] += -0.0671855890902924;
                        } else {
                          result[0] += -0.018503097503332632;
                        }
                      } else {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.825422286987305576) ) ) {
                          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                            if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                              if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                                result[0] += 0.036232175417086454;
                              } else {
                                result[0] += -0.07796239434695233;
                              }
                            } else {
                              result[0] += -0.07314870955907996;
                            }
                          } else {
                            result[0] += 0.022311232817068587;
                          }
                        } else {
                          result[0] += 0.018160073787476524;
                        }
                      }
                    } else {
                      result[0] += 0.0016786382459509468;
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)6.809154510498047763) ) ) {
                    result[0] += -0.03570942892872963;
                  } else {
                    result[0] += -0.1189058637044434;
                  }
                } else {
                  if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
                    result[0] += 0.04901136833376182;
                  } else {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)72.00000000000001421) ) ) {
                      result[0] += -0.06122639325809988;
                    } else {
                      result[0] += 0.05360125064163565;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)80.00000000000001421) ) ) {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.484580039978028232) ) ) {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.172047138214112216) ) ) {
                        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                          if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)1.500000000000000222) ) ) {
                            result[0] += 0.001619157840734999;
                          } else {
                            if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)2.500000000000000444) ) ) {
                              result[0] += -0.06605360213772923;
                            } else {
                              result[0] += 0.0017420001493828744;
                            }
                          }
                        } else {
                          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
                            result[0] += 0.18349990470839495;
                          } else {
                            result[0] += 0.01922512685569014;
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.500000000000000444) ) ) {
                          result[0] += -0.08043283951021796;
                        } else {
                          result[0] += 0.04018477422438853;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.869339942932130683) ) ) {
                        result[0] += -0.01665564692284137;
                      } else {
                        result[0] += 0.03141030395582101;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.673553824424744096) ) ) {
                      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                        result[0] += 0.018818515057095357;
                      } else {
                        result[0] += -0.052819068910348266;
                      }
                    } else {
                      if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += -0.044690039087266695;
                      } else {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.467917680740357333) ) ) {
                          result[0] += 0.013942250044965066;
                        } else {
                          result[0] += -0.035693253796285754;
                        }
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.484580039978028232) ) ) {
                    result[0] += -0.041429171648000564;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.834949493408204901) ) ) {
                      result[0] += -0.031475240978380496;
                    } else {
                      result[0] += 0.023347318584593643;
                    }
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)104.0000000000000142) ) ) {
              if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.024352178356277395;
                } else {
                  result[0] += -0.04761354820115974;
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.154959201812744585) ) ) {
                  result[0] += -0.07819822169624208;
                } else {
                  result[0] += 0.04672628229435402;
                }
              }
            } else {
              result[0] += -0.020213795738413212;
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)7.500000000000000888) ) ) {
    if ( UNLIKELY(  (data[46].missing != -1) && (data[46].fvalue <= (double)-1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)240501391360.0000305) ) ) {
        if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.947818994522095615) ) ) {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.013317857728321223;
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.673553824424744096) ) ) {
                result[0] += -0.0603626649235973;
              } else {
                if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.715336322784424716) ) ) {
                        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                          result[0] += 0.09961846361530086;
                        } else {
                          result[0] += -0.051246283991256227;
                        }
                      } else {
                        result[0] += -0.25646696906160077;
                      }
                    } else {
                      result[0] += 0.040548835622967044;
                    }
                  } else {
                    result[0] += -0.18288539694840178;
                  }
                } else {
                  result[0] += -0.004043423204760336;
                }
              }
            }
          } else {
            result[0] += 0.07784488361905796;
          }
        } else {
          if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
            result[0] += 0.006228419613058196;
          } else {
            result[0] += 0.03135841421728971;
          }
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.497866153717041238) ) ) {
          result[0] += 0.010275664038625726;
        } else {
          result[0] += 0.05737533323173685;
        }
      }
    } else {
      result[0] += -0.002271667325961405;
    }
  } else {
    if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)9.500000000000001776) ) ) {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
        result[0] += 0.056694823503298575;
      } else {
        if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)18.50000000000000355) ) ) {
          result[0] += 0.02366112410409831;
        } else {
          result[0] += -0.050999711970998764;
        }
      }
    } else {
      if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.357691764831543413) ) ) {
          result[0] += -0.05562787133566054;
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.154959201812744585) ) ) {
            result[0] += -0.03996929950989145;
          } else {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.617236852645874912) ) ) {
                if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.970608234405518466) ) ) {
                  result[0] += 0.034742821936280484;
                } else {
                  result[0] += -0.07430719369797203;
                }
              } else {
                result[0] += -0.0551064983584731;
              }
            } else {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.993164777755738193) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.993164777755738193) ) ) {
                    result[0] += -0.07944792172778538;
                  } else {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                      result[0] += -0.08976128534117755;
                    } else {
                      result[0] += 0.01043297336616467;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.464467763900757724) ) ) {
                    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.90474271774292081) ) ) {
                        result[0] += -0.1157353747295457;
                      } else {
                        result[0] += 0.12385805647439828;
                      }
                    } else {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                          result[0] += -0.07010930845039039;
                        } else {
                          result[0] += 0.005021257766870732;
                        }
                      } else {
                        result[0] += 0.06926776853510898;
                      }
                    }
                  } else {
                    result[0] += 0.05313800485998561;
                  }
                }
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.23832273483276456) ) ) {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.484580039978028232) ) ) {
                      result[0] += -0.04059443462654891;
                    } else {
                      result[0] += 0.026324119841694478;
                    }
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.208071470260621005) ) ) {
                      if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.970608234405518466) ) ) {
                        result[0] += 0.06851500143087524;
                      } else {
                        result[0] += -0.04728954012454498;
                      }
                    } else {
                      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                        result[0] += 0.0404596832554488;
                      } else {
                        result[0] += -0.023466982346546192;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.124530076980591708) ) ) {
                    result[0] += 0.010034399543524555;
                  } else {
                    result[0] += -0.05332114457367537;
                  }
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.284418344497681552) ) ) {
          if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)12.50000000000000178) ) ) {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.142630577087403232) ) ) {
                  result[0] += -0.03353338462972932;
                } else {
                  result[0] += 0.08822556779422586;
                }
              } else {
                if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)11.50000000000000178) ) ) {
                  result[0] += -0.07405346316850837;
                } else {
                  result[0] += 0.019181619916167365;
                }
              }
            } else {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                  result[0] += 0.021227900640917743;
                } else {
                  if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5.830332040786744052) ) ) {
                    result[0] += -0.025750099210723687;
                  } else {
                    result[0] += -0.10419080293207227;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.484580039978028232) ) ) {
                  if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.500000000000000444) ) ) {
                    result[0] += 0.03212192318756942;
                  } else {
                    result[0] += -0.026154672030671396;
                  }
                } else {
                  if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.04849097874315286;
                  } else {
                    if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)11.50000000000000178) ) ) {
                      result[0] += 0.0072653042569855225;
                    } else {
                      result[0] += 0.2960144524025365;
                    }
                  }
                }
              }
            }
          } else {
            result[0] += -0.0879930307445646;
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
            if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)5.500000000000000888) ) ) {
              if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)13.50000000000000178) ) ) {
                result[0] += -0.09585815340680456;
              } else {
                result[0] += 0.0012617865200212526;
              }
            } else {
              result[0] += 0.04926717878613551;
            }
          } else {
            if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                result[0] += -0.06794869360711614;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.029068946838379794) ) ) {
                  result[0] += 0.02226942111990848;
                } else {
                  result[0] += -0.05457180421261881;
                }
              }
            } else {
              if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.070054531097412998) ) ) {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += -0.053742613861846256;
                  } else {
                    result[0] += 0.06627813052418566;
                  }
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.029068946838379794) ) ) {
                    result[0] += -0.023689938371279473;
                  } else {
                    result[0] += 0.3189695608033867;
                  }
                }
              } else {
                result[0] += -0.08351381121547147;
              }
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)7.500000000000000888) ) ) {
    if ( UNLIKELY(  (data[27].missing != -1) && (data[27].fvalue <= (double)-1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)240501391360.0000305) ) ) {
        if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.255632162094117099) ) ) {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.010497860663851834;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.114721298217775214) ) ) {
                result[0] += -0.061157483758045285;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.357691764831543413) ) ) {
                  result[0] += -0.0019281045111489712;
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
                    if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.022419763186736493;
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.962127923965454546) ) ) {
                        result[0] += -0.009804820220862294;
                      } else {
                        result[0] += -0.14818704615096825;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.484580039978028232) ) ) {
                      result[0] += -0.11573534950244631;
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.397998809814454013) ) ) {
                        result[0] += 0.06965579888151008;
                      } else {
                        result[0] += -0.015146513385454552;
                      }
                    }
                  }
                }
              }
            }
          } else {
            result[0] += 0.06536809688533779;
          }
        } else {
          if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
            result[0] += 0.006231632086033707;
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.497866153717041238) ) ) {
              result[0] += 0.022191257445443868;
            } else {
              result[0] += 0.08658668063398618;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.497866153717041238) ) ) {
          result[0] += 0.010261842629953704;
        } else {
          result[0] += 0.05413789285954083;
        }
      }
    } else {
      if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
        if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)2.087193608283997026) ) ) {
          result[0] += -0.0024190220969107753;
        } else {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)120.0000000000000142) ) ) {
            result[0] += -0.08782714324358;
          } else {
            result[0] += 0.014681247786296285;
          }
        }
      } else {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.029068946838379794) ) ) {
          result[0] += -0.06262600422102926;
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.924581527709961826) ) ) {
              if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)240501391360.0000305) ) ) {
                result[0] += 0.00392056017105512;
              } else {
                result[0] += -0.07922787594301395;
              }
            } else {
              result[0] += 0.07281785360414227;
            }
          } else {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.08092154756190074;
            } else {
              if ( LIKELY( !(data[10].missing != -1) || (data[10].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                    result[0] += -0.07900641094521377;
                  } else {
                    result[0] += 0.09555061534850207;
                  }
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.607751369476319248) ) ) {
                    result[0] += 0.07245883290466425;
                  } else {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += 0.020270699809291005;
                    } else {
                      result[0] += -0.11593372548394623;
                    }
                  }
                }
              } else {
                result[0] += 0.10278977062175382;
              }
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)9.500000000000001776) ) ) {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.357691764831543413) ) ) {
        result[0] += 0.07235174652254246;
      } else {
        if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.500000000000000444) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
            result[0] += 0.05229650532555499;
          } else {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
              if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.059560208136456884;
              } else {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                  result[0] += -0.024068829940505486;
                } else {
                  result[0] += 0.08740361170929681;
                }
              }
            } else {
              if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.1079147565210399;
                } else {
                  result[0] += 0.045666373645460646;
                }
              } else {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.32928716091973653;
                } else {
                  result[0] += -0.07522915472550942;
                }
              }
            }
          }
        } else {
          result[0] += 0.033279493391673645;
        }
      }
    } else {
      if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
        if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)1.500000000000000222) ) ) {
          if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.651049375534058505) ) ) {
              if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.014434405214854945;
              } else {
                if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.142630577087403232) ) ) {
                    result[0] += -0.09368482186831635;
                  } else {
                    result[0] += 0.06105715218805008;
                  }
                } else {
                  result[0] += -0.06145480359708147;
                }
              }
            } else {
              result[0] += -0.04867267210591214;
            }
          } else {
            result[0] += -0.08632798717158235;
          }
        } else {
          if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)3.500000000000000444) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.1822080612182635) ) ) {
              result[0] += 0.3832534286305849;
            } else {
              result[0] += 0.05419191072248665;
            }
          } else {
            if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)4.500000000000000888) ) ) {
              if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                  result[0] += -0.07075658339242233;
                } else {
                  result[0] += 0.03919778348170705;
                }
              } else {
                result[0] += -0.08951567603440312;
              }
            } else {
              result[0] += 0.004609294488990325;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.694163918495178667) ) ) {
          result[0] += -0.021278267301073444;
        } else {
          if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)6.000000000000000888) ) ) {
            result[0] += -0.08209369034292613;
          } else {
            if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += 0.09851751980994936;
            } else {
              if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.022649366692593306;
                } else {
                  result[0] += -0.15541549720099823;
                }
              } else {
                if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.827801465988160068) ) ) {
                    result[0] += 0.06996378637454591;
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.342454433441162998) ) ) {
                      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
                        result[0] += 0.09503209760042294;
                      } else {
                        result[0] += -0.08312304149408586;
                      }
                    } else {
                      result[0] += -0.05599725485063067;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.357691764831543413) ) ) {
                    result[0] += -0.08940514262829011;
                  } else {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)104.0000000000000142) ) ) {
                      result[0] += 0.062299104055119764;
                    } else {
                      result[0] += -0.0044075630093920945;
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
  if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)7.500000000000000888) ) ) {
    if ( UNLIKELY(  (data[46].missing != -1) && (data[46].fvalue <= (double)-1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)240501391360.0000305) ) ) {
        if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.673553824424744096) ) ) {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.01569300590128552;
            } else {
              result[0] += -0.05619194305520592;
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.255632162094117099) ) ) {
              if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                result[0] += -0.0008274090782456828;
              } else {
                if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.02857420637807669;
                } else {
                  result[0] += -0.1262420796531599;
                }
              }
            } else {
              result[0] += 0.07210229999493473;
            }
          }
        } else {
          if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
            result[0] += 0.006117307931541015;
          } else {
            result[0] += 0.029312035742862164;
          }
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.497866153717041238) ) ) {
          result[0] += 0.009508252445211349;
        } else {
          result[0] += 0.05282342947073011;
        }
      }
    } else {
      if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
        if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)2.891912579536438432) ) ) {
          result[0] += -0.0026358774312936904;
        } else {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)232.0000000000000284) ) ) {
            result[0] += -0.0771597345500704;
          } else {
            result[0] += 0.0423776436780164;
          }
        }
      } else {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.029068946838379794) ) ) {
          result[0] += -0.06011040792814665;
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.695749998092652255) ) ) {
              if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)240501391360.0000305) ) ) {
                result[0] += 0.0018339694503903877;
              } else {
                result[0] += -0.07449282788760189;
              }
            } else {
              result[0] += 0.06001867117753721;
            }
          } else {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.08376134607232913;
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.208071470260621005) ) ) {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                    result[0] += -0.08185398757620681;
                  } else {
                    result[0] += 0.058702672921379866;
                  }
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.607751369476319248) ) ) {
                    result[0] += 0.06492540314842327;
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                      result[0] += -0.10695700664128856;
                    } else {
                      result[0] += 0.04414912945105486;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += 0.09963469937654726;
                } else {
                  result[0] += -0.03597834867741447;
                }
              }
            }
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
      if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)1.500000000000000222) ) ) {
        if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.651049375534058505) ) ) {
            if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
              result[0] += -0.015964993748017006;
            } else {
              result[0] += 0.03888493760425475;
            }
          } else {
            result[0] += -0.049791321287207105;
          }
        } else {
          result[0] += -0.08565920933946909;
        }
      } else {
        if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)3.500000000000000444) ) ) {
          if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)14.50000000000000178) ) ) {
            result[0] += 0.03013994395944002;
          } else {
            result[0] += 0.11631378672675104;
          }
        } else {
          if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)5.500000000000000888) ) ) {
            if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.516936540603638583) ) ) {
                result[0] += 0.001604681808956761;
              } else {
                result[0] += -0.08100231627202538;
              }
            } else {
              result[0] += -0.07011993761473702;
            }
          } else {
            if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)9.500000000000001776) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
                result[0] += 0.053333170211871575;
              } else {
                if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)15.50000000000000178) ) ) {
                  result[0] += 0.02525468962751133;
                } else {
                  result[0] += -0.03981073054212026;
                }
              }
            } else {
              if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.827801465988160068) ) ) {
                    result[0] += 0.057821355680736534;
                  } else {
                    result[0] += -0.014081615188166814;
                  }
                } else {
                  result[0] += -0.04682174200194589;
                }
              } else {
                if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)17.50000000000000355) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                    result[0] += 0.00490505010812087;
                  } else {
                    if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.06702427158462261;
                    } else {
                      if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)12.50000000000000178) ) ) {
                        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.909855604171753818) ) ) {
                          result[0] += 0.006573264224416534;
                        } else {
                          result[0] += -0.06971349908975961;
                        }
                      } else {
                        result[0] += -0.0639206967831714;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.516936540603638583) ) ) {
                      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += -0.029072024912128753;
                        } else {
                          if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.029068946838379794) ) ) {
                                result[0] += -0.10037801313437877;
                              } else {
                                result[0] += -0.8500034128220023;
                              }
                            } else {
                              result[0] += -0.15927864089144617;
                            }
                          } else {
                            result[0] += -0.10400988836704742;
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                          result[0] += 0.06593430870075584;
                        } else {
                          result[0] += -0.04339049722929694;
                        }
                      }
                    } else {
                      result[0] += 0.008504534648832955;
                    }
                  } else {
                    if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)2.500000000000000444) ) ) {
                      result[0] += 0.033846899988750435;
                    } else {
                      result[0] += 0.0038238886549264526;
                    }
                  }
                }
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.694163918495178667) ) ) {
        result[0] += -0.018149660194934088;
      } else {
        if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
          result[0] += 0.08487233323227182;
        } else {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
            if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.01728406005613677;
                } else {
                  result[0] += -0.15688302803430398;
                }
              } else {
                result[0] += 0.0499428714375326;
              }
            } else {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += -0.050914082277919374;
              } else {
                result[0] += 0.022338115438485504;
              }
            }
          } else {
            result[0] += 0.05417194586261136;
          }
        }
      }
    }
  }
}

