
#include "header.h"

void predict_unit0(union Entry* data, double* result) {
  unsigned int tmp;
  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY(  (data[27].missing != -1) && (data[27].fvalue <= (double)-1.00000001800250948e-35) ) ) {
      result[0] += 0.19534294557339324;
    } else {
      if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)9.500000000000001776) ) ) {
        if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.778982400894165927) ) ) {
            if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.02348177773617623;
            } else {
              result[0] += 0.1345109525644882;
            }
          } else {
            if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.19083217795431986;
            } else {
              if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.10775478782410193;
              } else {
                result[0] += -0.177740972203575;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.778982400894165927) ) ) {
              result[0] += 0.18734964152829756;
            } else {
              result[0] += 0.08451616356918701;
            }
          } else {
            if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
              result[0] += -0.14031292133466639;
            } else {
              result[0] += 0.1275951590874156;
            }
          }
        }
      } else {
        result[0] += 0.180149827122771;
      }
    }
  } else {
    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
      if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
            if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)27.50000000000000355) ) ) {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)56.00000000000000711) ) ) {
                result[0] += -0.18205801979336436;
              } else {
                result[0] += -0.09126450409631598;
              }
            } else {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.851041555404663974) ) ) {
                result[0] += -0.15902350126241308;
              } else {
                result[0] += 0.12906709689912857;
              }
            }
          } else {
            if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
              result[0] += 0.13335652858351107;
            } else {
              result[0] += -0.03738248825283384;
            }
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.284418344497681552) ) ) {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.172047138214112216) ) ) {
              result[0] += -0.1571727042823316;
            } else {
              result[0] += 0.05012583081017905;
            }
          } else {
            if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
                result[0] += -0.19900452954795878;
              } else {
                if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)21.50000000000000355) ) ) {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)40.00000000000000711) ) ) {
                    if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
                      result[0] += 0.07072618184287484;
                    } else {
                      result[0] += -0.15267623052425205;
                    }
                  } else {
                    result[0] += 0.14727574778008828;
                  }
                } else {
                  result[0] += 0.1789748566788684;
                }
              }
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.484580039978028232) ) ) {
                result[0] += 0.08273855237083749;
              } else {
                result[0] += -0.16780865463735792;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.20590913295745894) ) ) {
          result[0] += 0.05522503953205389;
        } else {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
            result[0] += -0.19296345675039914;
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.516936540603638583) ) ) {
              result[0] += -0.17435330273459274;
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.700598716735840066) ) ) {
                result[0] += 0.08899437172650068;
              } else {
                result[0] += -0.1126233174646447;
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)112.0000000000000142) ) ) {
            if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.484580039978028232) ) ) {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += -0.10932191363442889;
                  } else {
                    result[0] += 0.020227647661376292;
                  }
                } else {
                  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                    result[0] += -0.18911617768679442;
                  } else {
                    result[0] += -0.037702517968631194;
                  }
                }
              } else {
                result[0] += 0.11427738568498128;
              }
            } else {
              if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                  if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)17.50000000000000355) ) ) {
                    result[0] += 0.15050349290558918;
                  } else {
                    result[0] += -0.13620951346918939;
                  }
                } else {
                  result[0] += -0.2;
                }
              } else {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += 0.07646922950024485;
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.484580039978028232) ) ) {
                    result[0] += 0.12781246957703227;
                  } else {
                    result[0] += -0.17224179487770908;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)120.0000000000000142) ) ) {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.484580039978028232) ) ) {
                  result[0] += -0.036772352703034784;
                } else {
                  result[0] += 0.09781360598555004;
                }
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.397998809814454013) ) ) {
                  if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.053136512326435736;
                  } else {
                    result[0] += 0.16262913430696596;
                  }
                } else {
                  if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                      result[0] += 0.14937026979868182;
                    } else {
                      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += 0.02746930611610391;
                      } else {
                        result[0] += -0.12246892407613577;
                      }
                    }
                  } else {
                    result[0] += 0.18366637260224672;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += -0.18246615148460374;
              } else {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
                  if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
                    result[0] += 0.07969928108637117;
                  } else {
                    result[0] += -0.06093306848276734;
                  }
                } else {
                  result[0] += 0.11399539928604015;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.651049375534058505) ) ) {
            if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)5.500000000000000888) ) ) {
              if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
                if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.17990471114567164;
                } else {
                  result[0] += -0.014552278136886438;
                }
              } else {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += -0.11822498952420524;
                } else {
                  result[0] += 0.09646325709982076;
                }
              }
            } else {
              result[0] += 0.1353250657391729;
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
              result[0] += 0.04971727029705075;
            } else {
              result[0] += -0.17499726537899463;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)16.50000000000000355) ) ) {
          result[0] += 0.1966784484972165;
        } else {
          result[0] += 0.07589961402625198;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
      result[0] += 0.16597555587356988;
    } else {
      result[0] += 0.10779410776199319;
    }
  } else {
    if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
            result[0] += -0.16908528893708716;
          } else {
            if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)22.50000000000000355) ) ) {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.208071470260621005) ) ) {
                  if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)19.50000000000000355) ) ) {
                    if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.500000000000000444) ) ) {
                      result[0] += -0.07120890529108789;
                    } else {
                      result[0] += 0.07071565011113821;
                    }
                  } else {
                    result[0] += 0.13311080922416096;
                  }
                } else {
                  result[0] += -0.11850001142642048;
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.778982400894165927) ) ) {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += 0.11566098010320561;
                  } else {
                    result[0] += -0.1026767075119056;
                  }
                } else {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)20.00000000000000355) ) ) {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
                      result[0] += 0.0034841486690601713;
                    } else {
                      result[0] += 0.11667691940019281;
                    }
                  } else {
                    result[0] += -0.05063173318350255;
                  }
                }
              }
            } else {
              result[0] += 0.15217114500965245;
            }
          }
        } else {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
            if ( LIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.040716171264650214) ) ) {
                  result[0] += -0.03142289683617857;
                } else {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += -0.162961622026863;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.719506263732911933) ) ) {
                      result[0] += -0.03453365904887527;
                    } else {
                      result[0] += -0.11939478567290572;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += -0.14907302102868808;
                } else {
                  result[0] += -0.007672609027401866;
                }
              }
            } else {
              if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)5.000000000000000888) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
                  result[0] += -0.09844371645083043;
                } else {
                  result[0] += 0.026972744057975496;
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.1822080612182635) ) ) {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)56.00000000000000711) ) ) {
                    result[0] += 0.14841218816858082;
                  } else {
                    result[0] += 0.055351349575064074;
                  }
                } else {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += -0.09865586811004406;
                  } else {
                    result[0] += 0.03342970346518322;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)72.00000000000001421) ) ) {
              if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)11.50000000000000178) ) ) {
                if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)6.500000000000000888) ) ) {
                  if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)112.0000000000000142) ) ) {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.142630577087403232) ) ) {
                      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                        result[0] += 0.053247499546464905;
                      } else {
                        result[0] += -0.07054531067372126;
                      }
                    } else {
                      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)20.00000000000000355) ) ) {
                        if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)2.500000000000000444) ) ) {
                          result[0] += 0.004128416571983793;
                        } else {
                          result[0] += -0.08463188932773534;
                        }
                      } else {
                        result[0] += -0.1350827460684561;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                      if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.475347042083742011) ) ) {
                          result[0] += 0.024836397219307423;
                        } else {
                          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                            if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)7.500000000000000888) ) ) {
                              if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                                result[0] += -0.06595612090371808;
                              } else {
                                result[0] += 0.15883245490884;
                              }
                            } else {
                              result[0] += -0.18720457375501334;
                            }
                          } else {
                            result[0] += -0.12181609346939998;
                          }
                        }
                      } else {
                        result[0] += 0.1089786000686728;
                      }
                    } else {
                      if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += -0.08108804241723773;
                      } else {
                        if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)6.000000000000000888) ) ) {
                          result[0] += -0.1697128318822503;
                        } else {
                          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
                            if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)5.000000000000000888) ) ) {
                              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                                if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                                  result[0] += 0.02003482090672353;
                                } else {
                                  result[0] += 0.11965340099801727;
                                }
                              } else {
                                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.040716171264650214) ) ) {
                                  result[0] += -0.04828080471499816;
                                } else {
                                  result[0] += 0.12010230127258106;
                                }
                              }
                            } else {
                              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                                result[0] += 0.016036421525597682;
                              } else {
                                result[0] += -0.17601269734921746;
                              }
                            }
                          } else {
                            result[0] += 0.12934762364419658;
                          }
                        }
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)8.500000000000001776) ) ) {
                    result[0] += 0.17005493032606153;
                  } else {
                    if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)9.500000000000001776) ) ) {
                      result[0] += -0.17677001606541223;
                    } else {
                      result[0] += 0.11062765445448473;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.284418344497681552) ) ) {
                  result[0] += 0.046723142220248315;
                } else {
                  result[0] += -0.15274170552281932;
                }
              }
            } else {
              result[0] += -0.12052011188322027;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.500000000000000444) ) ) {
            result[0] += -0.08007087191406374;
          } else {
            result[0] += 0.05148437235654414;
          }
        } else {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += -0.16075942928804848;
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.484580039978028232) ) ) {
              if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)5.500000000000000888) ) ) {
                if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
                  result[0] += -0.13894713869020237;
                } else {
                  result[0] += -0.010429164272103263;
                }
              } else {
                result[0] += 0.05969766761284824;
              }
            } else {
              result[0] += -0.15604206185191438;
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)16.50000000000000355) ) ) {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.051912069320679599) ) ) {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.1673580792545438;
            } else {
              result[0] += -0.08125902781172284;
            }
          } else {
            result[0] += 0.12250306946003142;
          }
        } else {
          result[0] += 0.16063482740764506;
        }
      } else {
        result[0] += -0.1529084570270702;
      }
    }
  }
  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
        result[0] += 0.14725197226130413;
      } else {
        if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += 0.14829078062461365;
        } else {
          result[0] += 0.07333009197224936;
        }
      }
    } else {
      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
        if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)9.500000000000001776) ) ) {
          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)56.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.605039834976196733) ) ) {
              result[0] += -0.03467209596121243;
            } else {
              result[0] += -0.14401156282349897;
            }
          } else {
            result[0] += 0.05121638533054762;
          }
        } else {
          if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)5.500000000000000888) ) ) {
            result[0] += -0.15646714462595576;
          } else {
            result[0] += 0.10268112965602016;
          }
        }
      } else {
        if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)6.000000000000000888) ) ) {
          result[0] += -0.03949334275600231;
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.484580039978028232) ) ) {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += 0.09438097167340224;
            } else {
              result[0] += -0.06925262748208831;
            }
          } else {
            result[0] += 0.04554885157795244;
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
          if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)21.50000000000000355) ) ) {
            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.20590913295745894) ) ) {
              result[0] += 0.03293156980795533;
            } else {
              if ( LIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.1492936492136166;
              } else {
                if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)56.00000000000000711) ) ) {
                  result[0] += -0.13132606789835513;
                } else {
                  result[0] += -0.06245713940146628;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
              result[0] += -0.14424195661217015;
            } else {
              result[0] += 0.05463294932021001;
            }
          }
        } else {
          if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)40.00000000000000711) ) ) {
              if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)25.50000000000000355) ) ) {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                  result[0] += -0.11354232954316157;
                } else {
                  if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)10.50000000000000178) ) ) {
                    if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)5.500000000000000888) ) ) {
                      result[0] += -0.08473110000686197;
                    } else {
                      result[0] += 0.09487183274841587;
                    }
                  } else {
                    result[0] += -0.12668403050095542;
                  }
                }
              } else {
                result[0] += 0.10194050929973585;
              }
            } else {
              result[0] += 0.0649653651115009;
            }
          } else {
            if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.475347042083742011) ) ) {
                result[0] += -0.03014874810629772;
              } else {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)56.00000000000000711) ) ) {
                  result[0] += -0.13791388788601974;
                } else {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                    result[0] += 0.02710102569797773;
                  } else {
                    result[0] += -0.10927481357477925;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)16.50000000000000355) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.051912069320679599) ) ) {
                  result[0] += -0.16312988958096555;
                } else {
                  result[0] += 0.09750615318250289;
                }
              } else {
                result[0] += -0.1589500456959059;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
          if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
            if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
              result[0] += 0.09303474286895595;
            } else {
              result[0] += -0.0803090306107817;
            }
          } else {
            result[0] += -0.11503546374857232;
          }
        } else {
          result[0] += 0.1154750811260334;
        }
      }
    } else {
      if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)5.500000000000000888) ) ) {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)88.00000000000001421) ) ) {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)112.0000000000000142) ) ) {
              if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)4.500000000000000888) ) ) {
                if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.888826131820679155) ) ) {
                    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += 0.019558813099531572;
                    } else {
                      result[0] += -0.061414033973210484;
                    }
                  } else {
                    if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += -0.13617035756175577;
                    } else {
                      result[0] += 0.022532527882929768;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
                    result[0] += 0.020357555127791416;
                  } else {
                    result[0] += -0.10355242410691634;
                  }
                }
              } else {
                if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)17.50000000000000355) ) ) {
                  if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                    result[0] += 0.09610423651823086;
                  } else {
                    result[0] += -0.15983552476652854;
                  }
                } else {
                  result[0] += -0.1366719155272459;
                }
              }
            } else {
              if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)5.000000000000000888) ) ) {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                    result[0] += 0.10723148981234985;
                  } else {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.718933820724488193) ) ) {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.700598716735840066) ) ) {
                        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                          result[0] += -0.01712175080237542;
                        } else {
                          result[0] += 0.07904181537881003;
                        }
                      } else {
                        result[0] += 0.017496908156389667;
                      }
                    } else {
                      result[0] += -0.10610715643871599;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += -0.14595266277886715;
                  } else {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += 0.056813786231636444;
                    } else {
                      result[0] += -0.07467445992299399;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.700598716735840066) ) ) {
                  result[0] += 0.00038101671223415693;
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.232423543930054599) ) ) {
                    result[0] += -0.012543733758281364;
                  } else {
                    result[0] += 0.13111428545068457;
                  }
                }
              }
            }
          } else {
            result[0] += -0.10436107075396678;
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.284418344497681552) ) ) {
            if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)11.50000000000000178) ) ) {
              result[0] += 0.055566733670374326;
            } else {
              result[0] += -0.07665115841661135;
            }
          } else {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += -0.14293042225324898;
            } else {
              result[0] += 0.01601381816692293;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)16.50000000000000355) ) ) {
          if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.595119953155518466) ) ) {
            result[0] += 0.13596981264290223;
          } else {
            result[0] += 0.04115628407586834;
          }
        } else {
          result[0] += -0.16017369912435606;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
        result[0] += 0.13410821385887906;
      } else {
        if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += 0.13680928498403513;
        } else {
          result[0] += 0.0446585601721404;
        }
      }
    } else {
      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
        if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)9.500000000000001776) ) ) {
          if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)2.500000000000000444) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.605039834976196733) ) ) {
              result[0] += -0.03490589035746063;
            } else {
              result[0] += -0.13053031426571302;
            }
          } else {
            result[0] += 0.05174046612047187;
          }
        } else {
          if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)6.500000000000000888) ) ) {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.208071470260621005) ) ) {
              result[0] += 0.07797586466661754;
            } else {
              result[0] += -0.11367009297381017;
            }
          } else {
            result[0] += 0.1006737209495;
          }
        }
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.484580039978028232) ) ) {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)88.00000000000001421) ) ) {
            if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += -0.0559442626453959;
            } else {
              if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)7.000000000000000888) ) ) {
                result[0] += 0.10952021955907537;
              } else {
                if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                  result[0] += 0.11271448388663617;
                } else {
                  result[0] += -0.14072153955800637;
                }
              }
            }
          } else {
            result[0] += 0.010622822952623678;
          }
        } else {
          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)56.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += -0.07366612650557557;
            } else {
              result[0] += 0.05735777826278898;
            }
          } else {
            if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)7.000000000000000888) ) ) {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                result[0] += 0.07199300956659337;
              } else {
                result[0] += -0.05916465813821195;
              }
            } else {
              result[0] += -0.09052252618719675;
            }
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
        if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)29.50000000000000355) ) ) {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
            if ( LIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.49770236015319913) ) ) {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += -0.09437423119983099;
                } else {
                  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
                    result[0] += 0.1601885732140363;
                  } else {
                    result[0] += -0.16832735900587859;
                  }
                }
              } else {
                if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)21.50000000000000355) ) ) {
                  result[0] += -0.1342164331151421;
                } else {
                  if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                    result[0] += 0.014656388890543782;
                  } else {
                    result[0] += -0.1395402969981422;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)7.000000000000000888) ) ) {
                    result[0] += -0.08598940300629862;
                  } else {
                    result[0] += 0.044059020536021676;
                  }
                } else {
                  result[0] += -0.11840554243295838;
                }
              } else {
                if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)40.00000000000000711) ) ) {
                  result[0] += -0.12792540987613987;
                } else {
                  result[0] += 0.049860476487510824;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)10.50000000000000178) ) ) {
              if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.500000000000000444) ) ) {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)40.00000000000000711) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.475347042083742011) ) ) {
                    result[0] += -0.02738098435646577;
                  } else {
                    result[0] += -0.11939431459293426;
                  }
                } else {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                    result[0] += 0.049523571015423956;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.475347042083742011) ) ) {
                      result[0] += 0.0702983529424182;
                    } else {
                      result[0] += -0.09664267725872483;
                    }
                  }
                }
              } else {
                result[0] += 0.0589671442704946;
              }
            } else {
              result[0] += -0.1359966287102474;
            }
          }
        } else {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
            result[0] += -0.12941799077270855;
          } else {
            result[0] += 0.09057485724053224;
          }
        }
      } else {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
          if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
            result[0] += 0.0423583520400244;
          } else {
            result[0] += -0.11511513637512057;
          }
        } else {
          result[0] += 0.10098798522972074;
        }
      }
    } else {
      if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)5.500000000000000888) ) ) {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)88.00000000000001421) ) ) {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)112.0000000000000142) ) ) {
              if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)4.500000000000000888) ) ) {
                if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)2.500000000000000444) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.142630577087403232) ) ) {
                    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                      result[0] += 0.0074411549549561334;
                    } else {
                      result[0] += -0.06571994989283085;
                    }
                  } else {
                    result[0] += -0.14092688755332955;
                  }
                } else {
                  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
                    result[0] += 0.008528205358154213;
                  } else {
                    result[0] += -0.1056792847583161;
                  }
                }
              } else {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                  result[0] += 0.07080167880781933;
                } else {
                  result[0] += -0.14449161599220292;
                }
              }
            } else {
              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.20590913295745894) ) ) {
                result[0] += -0.131497584798045;
              } else {
                if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.302512168884278232) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.30829524993896662) ) ) {
                      result[0] += 0.07093670904805446;
                    } else {
                      result[0] += 0.020352281286658405;
                    }
                  } else {
                    result[0] += -0.05323799122268994;
                  }
                } else {
                  result[0] += 0.07558830072597719;
                }
              }
            }
          } else {
            result[0] += -0.09990342803603633;
          }
        } else {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.342454433441162998) ) ) {
            if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)11.50000000000000178) ) ) {
              result[0] += 0.025117612354371016;
            } else {
              result[0] += -0.09876028486789727;
            }
          } else {
            if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
              result[0] += -0.13351862194020067;
            } else {
              result[0] += -0.0019243794711579013;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)16.50000000000000355) ) ) {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.744781017303467685) ) ) {
              result[0] += -0.06182859944356154;
            } else {
              result[0] += 0.09997492246767019;
            }
          } else {
            result[0] += 0.1299713203715218;
          }
        } else {
          result[0] += -0.14373276701446155;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
        result[0] += 0.12527130298935932;
      } else {
        result[0] += 0.06937762339574231;
      }
    } else {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.484580039978028232) ) ) {
        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
            if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)5.500000000000000888) ) ) {
              result[0] += -0.008659311747959827;
            } else {
              result[0] += 0.10093033249343833;
            }
          } else {
            result[0] += 0.09132618918784527;
          }
        } else {
          result[0] += -0.05125434096142714;
        }
      } else {
        if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)6.500000000000000888) ) ) {
          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
            if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)19.50000000000000355) ) ) {
              result[0] += -0.1285372611750911;
            } else {
              result[0] += -0.002655678773213404;
            }
          } else {
            if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
              if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                  result[0] += 0.07182566087687595;
                } else {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += 0.08702783976186707;
                  } else {
                    result[0] += -0.10063684747366207;
                  }
                }
              } else {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)56.00000000000000711) ) ) {
                  result[0] += 0.047980078771819396;
                } else {
                  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)7.000000000000000888) ) ) {
                    result[0] += 0.032556698482752264;
                  } else {
                    result[0] += -0.09378738669294005;
                  }
                }
              }
            } else {
              result[0] += -0.10401962911884549;
            }
          }
        } else {
          if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)9.500000000000001776) ) ) {
            result[0] += -0.1306333332763143;
          } else {
            if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.13370646904281455;
            } else {
              if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)26.50000000000000355) ) ) {
                result[0] += -0.011865187231168764;
              } else {
                result[0] += 0.12364340978084294;
              }
            }
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
        if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)29.50000000000000355) ) ) {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.49770236015319913) ) ) {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += -0.1225861090810274;
              } else {
                result[0] += 0.08125423700366849;
              }
            } else {
              if ( LIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.12521083963188348;
              } else {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                    result[0] += -0.054879546130105455;
                  } else {
                    result[0] += -0.12434113029496785;
                  }
                } else {
                  result[0] += 0.012383686638154565;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)10.50000000000000178) ) ) {
              if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.500000000000000444) ) ) {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)40.00000000000000711) ) ) {
                  result[0] += -0.09430249517160323;
                } else {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                    result[0] += 0.0411387462670087;
                  } else {
                    result[0] += -0.05425791674997754;
                  }
                }
              } else {
                result[0] += 0.05134502218510009;
              }
            } else {
              result[0] += -0.12638129021434677;
            }
          }
        } else {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
            result[0] += -0.12171851424099295;
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.651049375534058505) ) ) {
              result[0] += -0.06660444280722333;
            } else {
              result[0] += 0.09361246636935036;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
          if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
            result[0] += 0.0326965424854944;
          } else {
            result[0] += -0.09124236016942429;
          }
        } else {
          result[0] += 0.08609819736517776;
        }
      }
    } else {
      if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)5.500000000000000888) ) ) {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)72.00000000000001421) ) ) {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)112.0000000000000142) ) ) {
              if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)4.500000000000000888) ) ) {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                  result[0] += -0.10803363812895697;
                } else {
                  if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)2.500000000000000444) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
                      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                        result[0] += 0.007214442351529008;
                      } else {
                        result[0] += -0.059925066629384606;
                      }
                    } else {
                      result[0] += -0.12440297634776468;
                    }
                  } else {
                    result[0] += 0.007248926998415686;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)6.500000000000000888) ) ) {
                  result[0] += 0.01677795825660435;
                } else {
                  if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)8.500000000000001776) ) ) {
                    result[0] += 0.10188816737748273;
                  } else {
                    if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)9.500000000000001776) ) ) {
                      result[0] += -0.158306270874169;
                    } else {
                      if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)17.50000000000000355) ) ) {
                        result[0] += 0.06960242334746783;
                      } else {
                        result[0] += -0.13090911704040586;
                      }
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
                  result[0] += -0.02439780216275911;
                } else {
                  result[0] += 0.08334340740685589;
                }
              } else {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
                  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)5.000000000000000888) ) ) {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                        result[0] += 0.08781395224559742;
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.69332504272461115) ) ) {
                          result[0] += 0.05050015756972548;
                        } else {
                          result[0] += -0.01942715027587424;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.302512168884278232) ) ) {
                        result[0] += 0.0976881762442478;
                      } else {
                        result[0] += 0.012044201181283687;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.467917680740357333) ) ) {
                        result[0] += 0.08373726498534523;
                      } else {
                        result[0] += -0.05714014547781768;
                      }
                    } else {
                      result[0] += -0.14746441291254891;
                    }
                  }
                } else {
                  result[0] += 0.09986039434005935;
                }
              }
            }
          } else {
            result[0] += -0.08846163239604823;
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
            result[0] += 0.0011183305140634054;
          } else {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += -0.11988574020992182;
            } else {
              result[0] += 5.686532903619896e-05;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)16.50000000000000355) ) ) {
          result[0] += 0.1160910464095071;
        } else {
          result[0] += -0.1314189608874173;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.497866153717041238) ) ) {
          result[0] += 0.032501868631038336;
        } else {
          result[0] += 0.11713356513541595;
        }
      } else {
        result[0] += 0.12785666056358758;
      }
    } else {
      if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
        result[0] += 0.1188880128050919;
      } else {
        result[0] += 0.03875171572276666;
      }
    }
  } else {
    if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)7.500000000000000888) ) ) {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)80.00000000000001421) ) ) {
            if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)3.500000000000000444) ) ) {
              if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.515218973159790483) ) ) {
                  result[0] += -0.0003451352744119143;
                } else {
                  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.142630577087403232) ) ) {
                      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                        if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)2.500000000000000444) ) ) {
                          if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                            result[0] += -0.14502873503927488;
                          } else {
                            result[0] += -0.035257334727740564;
                          }
                        } else {
                          result[0] += 0.10350585753274372;
                        }
                      } else {
                        result[0] += -0.09552590331903017;
                      }
                    } else {
                      if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)3.000000000000000444) ) ) {
                        result[0] += -0.042252727097070314;
                      } else {
                        result[0] += -0.12796617798316115;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.97438240051269709) ) ) {
                      result[0] += -0.0031607904277235425;
                    } else {
                      result[0] += -0.0863037855307005;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += -0.04618549139615699;
                } else {
                  if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                    result[0] += 0.04876810589706974;
                  } else {
                    result[0] += -0.011984136637278113;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)29.50000000000000355) ) ) {
                if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)8.500000000000001776) ) ) {
                  if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)6.500000000000000888) ) ) {
                    result[0] += -0.00803107916238122;
                  } else {
                    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                      if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                        result[0] += 0.09174458441510304;
                      } else {
                        result[0] += 0.03317043936561957;
                      }
                    } else {
                      result[0] += -0.1367437984546536;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)9.500000000000001776) ) ) {
                    result[0] += -0.10569462816570817;
                  } else {
                    if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)17.50000000000000355) ) ) {
                      result[0] += 0.04413918793406707;
                    } else {
                      if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                        result[0] += -0.1291896761335623;
                      } else {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.208071470260621005) ) ) {
                          result[0] += 0.038977583804323986;
                        } else {
                          if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)6.500000000000000888) ) ) {
                            result[0] += -0.13002174630175642;
                          } else {
                            result[0] += -0.017630637602283946;
                          }
                        }
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                  result[0] += -0.11730304597189867;
                } else {
                  result[0] += 0.08381418189227036;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)120.0000000000000142) ) ) {
              if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.04903092196267419;
              } else {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
                  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)7.000000000000000888) ) ) {
                    if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.039720773696899636) ) ) {
                        if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.500000000000000444) ) ) {
                          result[0] += 0.01181390427031777;
                        } else {
                          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                            result[0] += 0.10001165073424963;
                          } else {
                            result[0] += 0.011250492060636923;
                          }
                        }
                      } else {
                        result[0] += -0.06172756729880551;
                      }
                    } else {
                      if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                          result[0] += 0.12531635793685955;
                        } else {
                          result[0] += 0.06241568924809363;
                        }
                      } else {
                        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.778982400894165927) ) ) {
                          result[0] += 0.049770355787321294;
                        } else {
                          result[0] += 0.009789574780858602;
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                      result[0] += 0.026286446052013236;
                    } else {
                      result[0] += -0.13296467883453272;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.700598716735840066) ) ) {
                    result[0] += -0.023821643634756248;
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.342454433441162998) ) ) {
                      result[0] += -0.026504709845104335;
                    } else {
                      result[0] += 0.10056605698431412;
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)7.500000000000000888) ) ) {
                if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                  result[0] += -0.055175013060047184;
                } else {
                  result[0] += -0.12138954184193058;
                }
              } else {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
                  if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                    if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
                      result[0] += 0.01705136516181296;
                    } else {
                      result[0] += -0.0719614972445619;
                    }
                  } else {
                    result[0] += -0.07978239991114092;
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.029068946838379794) ) ) {
                    result[0] += -0.04283442318544343;
                  } else {
                    if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.172047138214112216) ) ) {
                      result[0] += 0.020733382186796098;
                    } else {
                      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)80.00000000000001421) ) ) {
                        result[0] += -0.023549533522044655;
                      } else {
                        result[0] += 0.11504061980786034;
                      }
                    }
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
            result[0] += -0.01568753449939149;
          } else {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.445705175399781162) ) ) {
              result[0] += 0.02951032672138413;
            } else {
              result[0] += -0.11579980120301539;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.484580039978028232) ) ) {
            if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)5.500000000000000888) ) ) {
              result[0] += -0.11081233232189491;
            } else {
              result[0] += -0.02582235425117408;
            }
          } else {
            result[0] += -0.12187714235185393;
          }
        } else {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
            result[0] += -0.15718015196910085;
          } else {
            result[0] += -0.04097167231030152;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)16.50000000000000355) ) ) {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.154959201812744585) ) ) {
          result[0] += 0.031549893622717075;
        } else {
          result[0] += 0.10418565564120295;
        }
      } else {
        result[0] += -0.13038275419307765;
      }
    }
  }
  if ( UNLIKELY(  (data[27].missing != -1) && (data[27].fvalue <= (double)-1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.497866153717041238) ) ) {
          result[0] += 0.02941639729113568;
        } else {
          result[0] += 0.11051099498047895;
        }
      } else {
        result[0] += 0.12652941637490908;
      }
    } else {
      if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
        result[0] += 0.11075559927431311;
      } else {
        result[0] += 0.014976583853398862;
      }
    }
  } else {
    if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)80.00000000000001421) ) ) {
          if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.778982400894165927) ) ) {
                if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.284418344497681552) ) ) {
                    if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)4.500000000000000888) ) ) {
                      if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
                        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.888826131820679155) ) ) {
                          result[0] += -0.03477926339352476;
                        } else {
                          result[0] += -0.12807485653884407;
                        }
                      } else {
                        result[0] += 0.015076504392254844;
                      }
                    } else {
                      result[0] += -0.10133035030103373;
                    }
                  } else {
                    result[0] += -0.11436499091156403;
                  }
                } else {
                  result[0] += -0.025137326950984796;
                }
              } else {
                if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.605039834976196733) ) ) {
                  result[0] += -0.11638706293832637;
                } else {
                  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                    result[0] += -0.11951544628119624;
                  } else {
                    result[0] += 0.02202260647047502;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.718933820724488193) ) ) {
                  result[0] += -0.029365128098565785;
                } else {
                  result[0] += -0.10595946051668899;
                }
              } else {
                result[0] += 0.025582739491660924;
              }
            }
          } else {
            if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)29.50000000000000355) ) ) {
              if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)8.500000000000001776) ) ) {
                if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)6.500000000000000888) ) ) {
                  if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2202.000000000000455) ) ) {
                    result[0] += -0.1276073959989544;
                  } else {
                    result[0] += -0.006693625987927738;
                  }
                } else {
                  if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                    result[0] += 0.06385459844808884;
                  } else {
                    result[0] += -0.12830729040550537;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)9.500000000000001776) ) ) {
                  result[0] += -0.10797942341587133;
                } else {
                  if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)17.50000000000000355) ) ) {
                    result[0] += 0.03615471330324736;
                  } else {
                    if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                      result[0] += -0.12095683555338677;
                    } else {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.208071470260621005) ) ) {
                        result[0] += 0.030295854752754505;
                      } else {
                        result[0] += -0.0710274116993168;
                      }
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                result[0] += -0.11349843016863247;
              } else {
                result[0] += 0.07337985682317548;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)88.00000000000001421) ) ) {
            if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.718933820724488193) ) ) {
                  result[0] += -0.04234686862668752;
                } else {
                  result[0] += -0.10706711214848552;
                }
              } else {
                result[0] += 0.06363313724997831;
              }
            } else {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
                if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)7.000000000000000888) ) ) {
                  if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.039720773696899636) ) ) {
                      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                        if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.500000000000000444) ) ) {
                          result[0] += 0.014035612892701442;
                        } else {
                          result[0] += 0.09103987666230728;
                        }
                      } else {
                        result[0] += 0.015265992937027165;
                      }
                    } else {
                      result[0] += -0.06090288075748987;
                    }
                  } else {
                    if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                      result[0] += 0.08516236430006133;
                    } else {
                      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.029068946838379794) ) ) {
                        result[0] += 0.04876135490008967;
                      } else {
                        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.067782521247864214) ) ) {
                          result[0] += 0.046979983917520514;
                        } else {
                          if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                            result[0] += -0.11595870734743817;
                          } else {
                            result[0] += 0.009256569408350237;
                          }
                        }
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                    result[0] += 0.020324649899562394;
                  } else {
                    result[0] += -0.12371823461962633;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.700598716735840066) ) ) {
                  result[0] += -0.02703496970803633;
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.342454433441162998) ) ) {
                    result[0] += -0.023474862633934033;
                  } else {
                    result[0] += 0.0862246761378649;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.302512168884278232) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                result[0] += -0.11575674611855512;
              } else {
                if ( LIKELY(  (data[44].missing != -1) && (data[44].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                  result[0] += -0.04197005273210872;
                } else {
                  result[0] += -0.12572373728967842;
                }
              }
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
                if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                  result[0] += -0.07590046425078961;
                } else {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += 0.012828190638528518;
                  } else {
                    result[0] += -0.07079782537500062;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)5.500000000000000888) ) ) {
                  result[0] += -0.07124198174222551;
                } else {
                  if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.918272972106934482) ) ) {
                    result[0] += 0.013709552491712993;
                  } else {
                    result[0] += 0.0767473999344695;
                  }
                }
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.484580039978028232) ) ) {
            if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)5.500000000000000888) ) ) {
              result[0] += -0.1038827890206051;
            } else {
              result[0] += -0.033570858455474736;
            }
          } else {
            result[0] += -0.11702035131437041;
          }
        } else {
          if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
            result[0] += -0.13883403338224107;
          } else {
            result[0] += -0.04417756538357543;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)16.50000000000000355) ) ) {
        if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)17.50000000000000355) ) ) {
          result[0] += 0.059635052673778555;
        } else {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.605039834976196733) ) ) {
            result[0] += 0.04449189435048196;
          } else {
            result[0] += 0.10197750919946187;
          }
        }
      } else {
        result[0] += -0.12297470650780301;
      }
    }
  }
  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.497866153717041238) ) ) {
            result[0] += 0.02670984359521117;
          } else {
            result[0] += 0.10504763704105383;
          }
        } else {
          result[0] += 0.12032109348862681;
        }
      } else {
        result[0] += 0.05027343406473877;
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.651049375534058505) ) ) {
        if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += -0.07760142316196239;
        } else {
          if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.0923664805044075;
            } else {
              if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)5.500000000000000888) ) ) {
                result[0] += -0.031319980490220854;
              } else {
                result[0] += 0.08599950593566916;
              }
            }
          } else {
            if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
              result[0] += -0.06572092622157723;
            } else {
              result[0] += 0.05507946306838355;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)6.500000000000000888) ) ) {
          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
            if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)4.500000000000000888) ) ) {
              result[0] += -0.10535659117243595;
            } else {
              result[0] += -0.013980302046139585;
            }
          } else {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.48738741874694913) ) ) {
              if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)4.500000000000000888) ) ) {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += 0.0496534500904422;
                } else {
                  result[0] += -0.03896132441701905;
                }
              } else {
                result[0] += 0.05663031595840188;
              }
            } else {
              if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                result[0] += 0.07593416652292202;
              } else {
                result[0] += -0.0683215004292016;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)9.500000000000001776) ) ) {
            result[0] += -0.11745918261496985;
          } else {
            result[0] += 0.08494937695111822;
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
      if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
          result[0] += -0.11029664819599097;
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.651049375534058505) ) ) {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += -0.1168117968528833;
            } else {
              result[0] += -0.016681825441432204;
            }
          } else {
            if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
                result[0] += -0.12461369147946194;
              } else {
                result[0] += 0.051444840387255414;
              }
            } else {
              result[0] += -0.10292115860010338;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
          if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.100254535675049272) ) ) {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += -0.10537286581906802;
            } else {
              result[0] += 0.060506740493088476;
            }
          } else {
            if ( LIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.11691256962354823;
            } else {
              result[0] += -0.03642062874631002;
            }
          }
        } else {
          if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
              if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.019570795709783224;
              } else {
                result[0] += -0.11391276027307883;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.719506263732911933) ) ) {
                result[0] += -0.0017079131569216772;
              } else {
                result[0] += -0.09948964803923731;
              }
            }
          } else {
            result[0] += 0.041997895104513135;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)7.500000000000000888) ) ) {
          if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.467917680740357333) ) ) {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += 0.055848127295375666;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.69332504272461115) ) ) {
                    result[0] += -0.1570387552404826;
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
                      result[0] += 0.09509662756743224;
                    } else {
                      result[0] += -0.047175334618276686;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += 0.002305142430126513;
                } else {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                    result[0] += 0.0655361547926003;
                  } else {
                    result[0] += -0.10883762784720707;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.20590913295745894) ) ) {
                result[0] += -0.14260873224113446;
              } else {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                  result[0] += 0.06315603291212626;
                } else {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)112.0000000000000142) ) ) {
                    if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)7.500000000000000888) ) ) {
                      result[0] += -0.062193036223976865;
                    } else {
                      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
                        result[0] += -0.06263906313912075;
                      } else {
                        result[0] += 0.03961921526176697;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.95235633850097834) ) ) {
                        result[0] += 0.06134900833505641;
                      } else {
                        result[0] += -0.027836007728376605;
                      }
                    } else {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.827801465988160068) ) ) {
                        if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += 0.07893467595759877;
                        } else {
                          if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                            result[0] += 0.04948116965692294;
                          } else {
                            result[0] += -0.10567360881043808;
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.834949493408204901) ) ) {
                          result[0] += -0.06973037365583638;
                        } else {
                          if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                            result[0] += -0.06591448248542787;
                          } else {
                            result[0] += 0.04617000019885609;
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
              result[0] += -0.09579067734334201;
            } else {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += -0.01864899045809504;
              } else {
                result[0] += 0.07028605174333179;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
            result[0] += 0.015200770862978837;
          } else {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += -0.10465161853682346;
            } else {
              result[0] += -0.011571400693211175;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)16.50000000000000355) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.48918962478637873) ) ) {
            result[0] += 0.09841975567863331;
          } else {
            result[0] += -0.04449033302172472;
          }
        } else {
          result[0] += -0.11065549830653297;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.700598716735840066) ) ) {
          result[0] += 0.04713002454999767;
        } else {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
            result[0] += 0.10345462455279067;
          } else {
            result[0] += 0.04891895564514019;
          }
        }
      } else {
        result[0] += 0.11723643809853555;
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.651049375534058505) ) ) {
        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)40.00000000000000711) ) ) {
          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)20.00000000000000355) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.671854496002199042) ) ) {
              result[0] += -0.004813695871352559;
            } else {
              result[0] += 0.07596092341157767;
            }
          } else {
            result[0] += 0.006231701562691707;
          }
        } else {
          result[0] += 0.08646732950231846;
        }
      } else {
        if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)17.50000000000000355) ) ) {
          if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)7.500000000000000888) ) ) {
            if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.04425679309592552;
            } else {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                result[0] += -0.10290187835732995;
              } else {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += 0.03959542250429107;
                } else {
                  result[0] += -0.04508997250926774;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)2.500000000000000444) ) ) {
              result[0] += -0.08823689467927687;
            } else {
              result[0] += 0.08420321953352682;
            }
          }
        } else {
          if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)15.50000000000000178) ) ) {
            result[0] += 0.10568105388555511;
          } else {
            result[0] += -0.132401214043812;
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
        if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)29.50000000000000355) ) ) {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
            if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.100254535675049272) ) ) {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += -0.10199622304195975;
              } else {
                result[0] += 0.04775928156579128;
              }
            } else {
              if ( LIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += -0.11638551403587793;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.040716171264650214) ) ) {
                    result[0] += 0.006757138113488399;
                  } else {
                    result[0] += -0.09852087563614731;
                  }
                }
              } else {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += -0.08100663201105457;
                } else {
                  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)40.00000000000000711) ) ) {
                    result[0] += -0.1037768785221084;
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.302512168884278232) ) ) {
                      result[0] += -0.060737120861306705;
                    } else {
                      result[0] += 0.08066681810052007;
                    }
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)10.50000000000000178) ) ) {
              if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.500000000000000444) ) ) {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)40.00000000000000711) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.671854496002199042) ) ) {
                    result[0] += -0.008261617501910519;
                  } else {
                    result[0] += -0.08848550565366613;
                  }
                } else {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                    result[0] += 0.033198533814409;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.719506263732911933) ) ) {
                      result[0] += 0.03451173211575872;
                    } else {
                      result[0] += -0.07553273777375305;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)12.50000000000000178) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.744781017303467685) ) ) {
                    result[0] += -0.04664726239972317;
                  } else {
                    result[0] += 0.06964006338041213;
                  }
                } else {
                  result[0] += -0.11933990481317308;
                }
              }
            } else {
              result[0] += -0.10483517321120227;
            }
          }
        } else {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
            result[0] += -0.10180776961073142;
          } else {
            result[0] += 0.05727330834203645;
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.1822080612182635) ) ) {
          result[0] += 0.0713142099410685;
        } else {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
              result[0] += -0.11097989081395727;
            } else {
              result[0] += 0.005598456907806209;
            }
          } else {
            result[0] += 0.05928921301718768;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)7.500000000000000888) ) ) {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)88.00000000000001421) ) ) {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)112.0000000000000142) ) ) {
              if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)4.500000000000000888) ) ) {
                if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)2.500000000000000444) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.02604460716247603) ) ) {
                    result[0] += -0.022520142079381494;
                  } else {
                    result[0] += -0.10827880677649873;
                  }
                } else {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
                    result[0] += -0.06832981822265595;
                  } else {
                    if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
                      result[0] += 0.031152852084896182;
                    } else {
                      result[0] += -0.07952382655248413;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                  if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                    result[0] += 0.05256068775789843;
                  } else {
                    result[0] += -0.12078466851330803;
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.827801465988160068) ) ) {
                    result[0] += 0.059817172009305176;
                  } else {
                    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += 0.029074972472413065;
                    } else {
                      result[0] += -0.09495921504940957;
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.20590913295745894) ) ) {
                result[0] += -0.1364343418090594;
              } else {
                result[0] += 0.045452926951865;
              }
            }
          } else {
            if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += -0.10984311579275932;
            } else {
              result[0] += -0.047550692951516954;
            }
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
            if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.500000000000000444) ) ) {
              result[0] += 0.04654145129281872;
            } else {
              result[0] += -0.06330802264179444;
            }
          } else {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.605039834976196733) ) ) {
              result[0] += 0.01662437861167905;
            } else {
              result[0] += -0.10371566866301547;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)16.50000000000000355) ) ) {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.051912069320679599) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
                result[0] += -0.04893347884122434;
              } else {
                result[0] += 0.07090784493288736;
              }
            } else {
              result[0] += 0.0846794230576229;
            }
          } else {
            result[0] += 0.0986566284839697;
          }
        } else {
          result[0] += -0.10553371837044306;
        }
      }
    }
  }
  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.497866153717041238) ) ) {
              result[0] += 0.021840482100287148;
            } else {
              result[0] += 0.09484601234095175;
            }
          } else {
            if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
              result[0] += 0.11543558408709514;
            } else {
              result[0] += 0.054689552135540845;
            }
          }
        } else {
          result[0] += 0.04098044235953163;
        }
      } else {
        if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)7.500000000000000888) ) ) {
          if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
            if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.02604460716247603) ) ) {
                result[0] += -0.024450253389323596;
              } else {
                result[0] += -0.11974903478926897;
              }
            } else {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)120.0000000000000142) ) ) {
                if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)112.0000000000000142) ) ) {
                  if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)6.500000000000000888) ) ) {
                    if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)5.500000000000000888) ) ) {
                      if ( LIKELY(  (data[50].missing != -1) && (data[50].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.232423543930054599) ) ) {
                          result[0] += -0.060238419789901146;
                        } else {
                          result[0] += -0.11238046273260345;
                        }
                      } else {
                        result[0] += -0.016445664945720858;
                      }
                    } else {
                      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.1822080612182635) ) ) {
                          result[0] += -0.013415862389859298;
                        } else {
                          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                            result[0] += -0.08887988665462045;
                          } else {
                            result[0] += 0.030246233604326753;
                          }
                        }
                      } else {
                        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)56.00000000000000711) ) ) {
                          result[0] += 0.033466262358376894;
                        } else {
                          result[0] += -0.01062697735003381;
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)8.500000000000001776) ) ) {
                      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                        result[0] += 0.05589487795425207;
                      } else {
                        result[0] += -0.11742345941382988;
                      }
                    } else {
                      if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)25.50000000000000355) ) ) {
                        if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)8.500000000000001776) ) ) {
                          if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)9.500000000000001776) ) ) {
                            result[0] += -0.10047997919988966;
                          } else {
                            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.651049375534058505) ) ) {
                              result[0] += 0.039610009082208665;
                            } else {
                              if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)19.50000000000000355) ) ) {
                                result[0] += 0.03560064685460481;
                              } else {
                                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.827801465988160068) ) ) {
                                  result[0] += 0.071680912345205;
                                } else {
                                  result[0] += -0.07618465000423316;
                                }
                              }
                            }
                          }
                        } else {
                          result[0] += -0.10091570739950935;
                        }
                      } else {
                        if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                          result[0] += -0.10044247227182442;
                        } else {
                          result[0] += 0.05722169823166985;
                        }
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.0459329131842716;
                  } else {
                    if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
                      if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)7.000000000000000888) ) ) {
                        if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                            if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.500000000000000444) ) ) {
                              result[0] += 0.0054487773657686004;
                            } else {
                              result[0] += 0.07921500903378202;
                            }
                          } else {
                            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.232423543930054599) ) ) {
                              result[0] += 0.015128961095719507;
                            } else {
                              if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)4.500000000000000888) ) ) {
                                result[0] += -0.07228506940026656;
                              } else {
                                result[0] += 0.05427384495038931;
                              }
                            }
                          }
                        } else {
                          if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                            if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                              result[0] += 0.05796126766778941;
                            } else {
                              result[0] += 0.10629336726932231;
                            }
                          } else {
                            result[0] += 0.033928457762897234;
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                          result[0] += 0.01988229844129287;
                        } else {
                          result[0] += -0.11841145418770255;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
                        result[0] += -0.03806371664533799;
                      } else {
                        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.342454433441162998) ) ) {
                          result[0] += -0.016451176044053262;
                        } else {
                          result[0] += 0.07935950510704974;
                        }
                      }
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
                  if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.010793990097342214;
                  } else {
                    result[0] += -0.07147430545333737;
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.284418344497681552) ) ) {
                    result[0] += -0.05816587719071134;
                  } else {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.249904870986938921) ) ) {
                      result[0] += -0.0559129091768362;
                    } else {
                      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.339395284652710849) ) ) {
                        result[0] += 0.017852572079011548;
                      } else {
                        if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += -0.06106076118131287;
                        } else {
                          result[0] += 0.09308181523666795;
                        }
                      }
                    }
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.802696108818054643) ) ) {
              result[0] += -0.09735801492035916;
            } else {
              if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += -0.1126667882965865;
              } else {
                if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)120.0000000000000142) ) ) {
                    result[0] += 0.035355721930615626;
                  } else {
                    result[0] += -0.05538311302768916;
                  }
                } else {
                  result[0] += -0.05690580889145232;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.142630577087403232) ) ) {
            result[0] += -0.011494789034448089;
          } else {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.445705175399781162) ) ) {
              result[0] += 0.015868985903298848;
            } else {
              result[0] += -0.10474774483378453;
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.484580039978028232) ) ) {
          if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)5.500000000000000888) ) ) {
            result[0] += -0.08724291506926883;
          } else {
            result[0] += -0.02208173379301404;
          }
        } else {
          result[0] += -0.10657308336082914;
        }
      } else {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.484580039978028232) ) ) {
          result[0] += -0.032371736700998376;
        } else {
          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
            result[0] += -0.00663509733724954;
          } else {
            result[0] += -0.08335026763871085;
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)16.50000000000000355) ) ) {
      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.938867926597595659) ) ) {
        result[0] += -0.0011564743145955475;
      } else {
        result[0] += 0.0839393044075755;
      }
    } else {
      result[0] += -0.11271014125208234;
    }
  }
  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.736135363578796831) ) ) {
          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += 0.09022317779398009;
          } else {
            result[0] += 0.024028900529011687;
          }
        } else {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
            result[0] += 0.09492164421001686;
          } else {
            result[0] += 0.03538847430626278;
          }
        }
      } else {
        result[0] += 0.11088802532763826;
      }
    } else {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.484580039978028232) ) ) {
        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)80.00000000000001421) ) ) {
          if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += -0.07380427575951438;
          } else {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)112.0000000000000142) ) ) {
              if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
                if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)18.50000000000000355) ) ) {
                  result[0] += 0.02197462642632214;
                } else {
                  result[0] += 0.10282760141467305;
                }
              } else {
                result[0] += 0.09395955029676228;
              }
            } else {
              result[0] += 0.08564925978392218;
            }
          }
        } else {
          result[0] += -0.04540369645728131;
        }
      } else {
        if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)6.500000000000000888) ) ) {
          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
            if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)19.50000000000000355) ) ) {
              result[0] += -0.09697306879257928;
            } else {
              result[0] += -0.0043854564273195975;
            }
          } else {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.48738741874694913) ) ) {
              result[0] += 0.03185964015242782;
            } else {
              result[0] += -0.02999418360093561;
            }
          }
        } else {
          if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += 0.1070457548822746;
          } else {
            if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
              result[0] += -0.10266505887162541;
            } else {
              result[0] += 0.0779530165622279;
            }
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
        if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)29.50000000000000355) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.481347560882569248) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.516936540603638583) ) ) {
              result[0] += -0.06653948647078795;
            } else {
              result[0] += 0.018839609755165063;
            }
          } else {
            if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)5.000000000000000888) ) ) {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += -0.1110190251424871;
              } else {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)56.00000000000000711) ) ) {
                  result[0] += -0.09782416227260235;
                } else {
                  result[0] += -0.04660772148855062;
                }
              }
            } else {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += -0.08187813919224718;
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.302512168884278232) ) ) {
                  result[0] += -0.07026431466497185;
                } else {
                  result[0] += 0.09201760200607088;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
            result[0] += -0.09669555828491608;
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.651049375534058505) ) ) {
              result[0] += -0.07508366738089846;
            } else {
              result[0] += 0.06015864318208172;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.10290670394897639) ) ) {
          if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
            result[0] += 0.06708338977098278;
          } else {
            result[0] += -0.15829407305084667;
          }
        } else {
          result[0] += -0.024684616920526627;
        }
      }
    } else {
      if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)7.500000000000000888) ) ) {
          if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.20590913295745894) ) ) {
              result[0] += -0.12000211381536151;
            } else {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)112.0000000000000142) ) ) {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.058293167009448844;
                  } else {
                    result[0] += 0.026496138403120914;
                  }
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.232423543930054599) ) ) {
                    result[0] += -0.012230019319466275;
                  } else {
                    result[0] += -0.07982445971572535;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
                    result[0] += -0.05540642298645232;
                  } else {
                    result[0] += 0.04770000081723927;
                  }
                } else {
                  if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)2.500000000000000444) ) ) {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.284418344497681552) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.671854496002199042) ) ) {
                          result[0] += 0.04588554766722608;
                        } else {
                          result[0] += -0.04045269649637138;
                        }
                      } else {
                        if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += -0.01180750381467744;
                        } else {
                          result[0] += 0.057496460690327815;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.040716171264650214) ) ) {
                        result[0] += -0.019099693445639346;
                      } else {
                        result[0] += 0.05186877613502262;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.807813167572023261) ) ) {
                        result[0] += -0.06797370811787878;
                      } else {
                        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                          result[0] += 0.05347466309805361;
                        } else {
                          result[0] += -0.04508703283845319;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.94957673549652144) ) ) {
                        result[0] += 0.069335048448684;
                      } else {
                        result[0] += -0.10321806285797563;
                      }
                    }
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += -0.10109918714143232;
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.694163918495178667) ) ) {
                  result[0] += 0.04228277242718351;
                } else {
                  result[0] += -0.09980682894597291;
                }
              }
            } else {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += -0.02357161128671831;
              } else {
                result[0] += 0.053333963468460414;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += -0.06443224914573231;
            } else {
              result[0] += 0.04759413124088574;
            }
          } else {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.605039834976196733) ) ) {
              result[0] += 0.009303115704407878;
            } else {
              result[0] += -0.100611879515106;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)16.50000000000000355) ) ) {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.778982400894165927) ) ) {
              result[0] += -0.04755072260337583;
            } else {
              result[0] += 0.061133656891878385;
            }
          } else {
            result[0] += 0.08865624976007963;
          }
        } else {
          result[0] += -0.10117453262548592;
        }
      }
    }
  }
  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
          if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.994492053985595925) ) ) {
              result[0] += 0.022550660317258713;
            } else {
              result[0] += 0.08557179350301664;
            }
          } else {
            if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
              result[0] += 0.10978655258760346;
            } else {
              result[0] += 0.044503551991980854;
            }
          }
        } else {
          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += 0.08724862179244332;
          } else {
            result[0] += 0.02153324922915748;
          }
        }
      } else {
        if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)7.500000000000000888) ) ) {
          if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
            if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.888826131820679155) ) ) {
                result[0] += -0.022761600383015192;
              } else {
                result[0] += -0.11515148412752163;
              }
            } else {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)120.0000000000000142) ) ) {
                if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)112.0000000000000142) ) ) {
                  if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)6.500000000000000888) ) ) {
                    if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)5.500000000000000888) ) ) {
                      if ( UNLIKELY(  (data[42].missing != -1) && (data[42].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                        result[0] += -0.017444218367435423;
                      } else {
                        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.232423543930054599) ) ) {
                          result[0] += -0.055009268695643335;
                        } else {
                          result[0] += -0.10762579428411917;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.1822080612182635) ) ) {
                          result[0] += -0.010216943406434873;
                        } else {
                          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                            result[0] += -0.09036693031991766;
                          } else {
                            result[0] += 0.0225965518979529;
                          }
                        }
                      } else {
                        result[0] += 0.02497038532633496;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)8.500000000000001776) ) ) {
                      result[0] += 0.047764178284841245;
                    } else {
                      if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)29.50000000000000355) ) ) {
                        if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)9.500000000000001776) ) ) {
                          result[0] += -0.0996643306581092;
                        } else {
                          if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)17.50000000000000355) ) ) {
                            result[0] += 0.022899639163122845;
                          } else {
                            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.208071470260621005) ) ) {
                              if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                                result[0] += -0.11165379146639363;
                              } else {
                                result[0] += 0.02667314491746795;
                              }
                            } else {
                              result[0] += -0.08434409008629572;
                            }
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                          result[0] += -0.09433411088879808;
                        } else {
                          result[0] += 0.0491681577521749;
                        }
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.04746526922042681;
                  } else {
                    if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
                      if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)7.000000000000000888) ) ) {
                        if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.302512168884278232) ) ) {
                            result[0] += 0.012910701139961426;
                          } else {
                            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                              result[0] += 0.036847997117972896;
                            } else {
                              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.040716171264650214) ) ) {
                                result[0] += 0.024675144729905077;
                              } else {
                                result[0] += -0.08236721230720592;
                              }
                            }
                          }
                        } else {
                          if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                              result[0] += 0.10135888838930092;
                            } else {
                              result[0] += 0.05073481520432387;
                            }
                          } else {
                            result[0] += 0.028237244902236853;
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                          result[0] += 0.011164690354915411;
                        } else {
                          result[0] += -0.1188619918439994;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
                        result[0] += -0.040420997568525435;
                      } else {
                        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.342454433441162998) ) ) {
                          result[0] += -0.019440556807468434;
                        } else {
                          result[0] += 0.06982198723776235;
                        }
                      }
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
                  if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                    result[0] += -0.0770468980459513;
                  } else {
                    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
                      result[0] += 0.0089611717905513;
                    } else {
                      result[0] += -0.061440660081056864;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)7.500000000000000888) ) ) {
                    result[0] += -0.06863638848431126;
                  } else {
                    if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.339395284652710849) ) ) {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.484580039978028232) ) ) {
                        result[0] += -0.044904921443598395;
                      } else {
                        result[0] += 0.013712590948261034;
                      }
                    } else {
                      if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += -0.049715995174675856;
                      } else {
                        result[0] += 0.09348303796730553;
                      }
                    }
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.41211462020874201) ) ) {
              if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += -0.11058191101848612;
              } else {
                if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += 0.002325736022052274;
                } else {
                  result[0] += -0.10156082169510833;
                }
              }
            } else {
              result[0] += -0.09280357851288919;
            }
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
            result[0] += -0.0013922301085356177;
          } else {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.445705175399781162) ) ) {
              result[0] += 0.017901521486434615;
            } else {
              result[0] += -0.09919914849078207;
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
        if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)5.500000000000000888) ) ) {
          result[0] += -0.10067465332997182;
        } else {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.284418344497681552) ) ) {
            result[0] += 0.01039291323789808;
          } else {
            if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
              result[0] += -0.10225266572259462;
            } else {
              result[0] += -0.04313395713951852;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.357691764831543413) ) ) {
          result[0] += -0.02055142118214022;
        } else {
          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
            result[0] += -0.0021761142503023715;
          } else {
            result[0] += -0.07953902699865179;
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)16.50000000000000355) ) ) {
      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.605039834976196733) ) ) {
        result[0] += 0.006895898086952493;
      } else {
        if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)2.500000000000000444) ) ) {
          if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += 0.0606036108667417;
          } else {
            result[0] += 0.1055341112809531;
          }
        } else {
          result[0] += 0.06459956513688173;
        }
      }
    } else {
      result[0] += -0.10676278346219095;
    }
  }
  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)3.000000000000000444) ) ) {
        if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
            result[0] += 0.08185913157043835;
          } else {
            result[0] += 0.04161235363983929;
          }
        } else {
          result[0] += 0.10360062650857026;
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
            result[0] += 0.018447920048764284;
          } else {
            if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += 0.08522632915492331;
            } else {
              result[0] += 0.003369930818262844;
            }
          }
        } else {
          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += -0.11143400803179598;
          } else {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.56941866874694913) ) ) {
              result[0] += 0.03132320103211178;
            } else {
              result[0] += -0.031800575637174526;
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)19.50000000000000355) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.484580039978028232) ) ) {
          result[0] += 0.04976667631508521;
        } else {
          result[0] += -0.07667939401287549;
        }
      } else {
        result[0] += 0.06552504598111643;
      }
    }
  } else {
    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
      if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)56.00000000000000711) ) ) {
            result[0] += -0.10334028630918007;
          } else {
            if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
              result[0] += 0.003618626075186257;
            } else {
              result[0] += -0.08915593604851792;
            }
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.778982400894165927) ) ) {
            result[0] += -0.07542676995680996;
          } else {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)40.00000000000000711) ) ) {
              if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)29.50000000000000355) ) ) {
                result[0] += -0.06375631628277659;
              } else {
                result[0] += 0.047913257477723986;
              }
            } else {
              result[0] += 0.05884638290494435;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.481347560882569248) ) ) {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.827801465988160068) ) ) {
            result[0] += -0.0694215317005324;
          } else {
            if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += 0.07140419126576496;
            } else {
              result[0] += -0.026322440508979195;
            }
          }
        } else {
          result[0] += -0.10589674322847538;
        }
      }
    } else {
      if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)7.500000000000000888) ) ) {
          if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)104.0000000000000142) ) ) {
              if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)112.0000000000000142) ) ) {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.057554699697533844;
                  } else {
                    result[0] += 0.020979472239194115;
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.284418344497681552) ) ) {
                    result[0] += -0.008457800602279844;
                  } else {
                    result[0] += -0.07378408418641488;
                  }
                }
              } else {
                if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)5.000000000000000888) ) ) {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.48918962478637873) ) ) {
                        if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.055836200714113104) ) ) {
                            result[0] += 0.054999368790054265;
                          } else {
                            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                                result[0] += 0.01855956607803358;
                              } else {
                                result[0] += -0.04468823137813999;
                              }
                            } else {
                              result[0] += 0.03172033021359274;
                            }
                          }
                        } else {
                          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                            result[0] += -0.002381444922166923;
                          } else {
                            result[0] += 0.0725280052403736;
                          }
                        }
                      } else {
                        result[0] += -0.05598914133192671;
                      }
                    } else {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.302512168884278232) ) ) {
                        result[0] += 0.06681675206720393;
                      } else {
                        result[0] += -0.013946679548354544;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += -0.10707610150271184;
                    } else {
                      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                        result[0] += 0.0341132015597769;
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.651049375534058505) ) ) {
                          result[0] += 0.04525710430695977;
                        } else {
                          result[0] += -0.08881339215044157;
                        }
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.232423543930054599) ) ) {
                      result[0] += -0.028055753201147168;
                    } else {
                      result[0] += 0.08433913893715232;
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.938988685607911933) ) ) {
                      result[0] += -0.1158694987991028;
                    } else {
                      result[0] += 0.047648084338046105;
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += -0.10343045488509472;
              } else {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += -0.05732049278856058;
                } else {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.917405366897583452) ) ) {
                    result[0] += -0.038036309755483944;
                  } else {
                    result[0] += 0.04385967600044441;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
              result[0] += -0.09021228164023853;
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.736135363578796831) ) ) {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += -0.002316377432221984;
                } else {
                  result[0] += 0.06667570690688966;
                }
              } else {
                result[0] += -0.06181230352056645;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.484580039978028232) ) ) {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += -0.06664088076314978;
            } else {
              if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)13.50000000000000178) ) ) {
                result[0] += -0.0028448687138829796;
              } else {
                result[0] += 0.114673531482584;
              }
            }
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.605039834976196733) ) ) {
              result[0] += 0.00033668171318775194;
            } else {
              result[0] += -0.09728380601487722;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.718933820724488193) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.070054531097412998) ) ) {
              result[0] += -0.0756373890189539;
            } else {
              result[0] += 0.05029540725959685;
            }
          } else {
            result[0] += 0.059086404933629544;
          }
        } else {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.56849193572998225) ) ) {
            if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)16.50000000000000355) ) ) {
              result[0] += 0.0926343977653456;
            } else {
              result[0] += -0.08147773146763436;
            }
          } else {
            result[0] += -0.04903570002497885;
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
        if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)137422176256.0000153) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.497866153717041238) ) ) {
            result[0] += 0.033503101949220675;
          } else {
            result[0] += 0.08412686153858656;
          }
        } else {
          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += 0.07843753723203913;
          } else {
            if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
              result[0] += -0.008144170147779675;
            } else {
              result[0] += 0.07461092924574003;
            }
          }
        }
      } else {
        result[0] += 0.10327204344731511;
      }
    } else {
      if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)17.50000000000000355) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
          if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += -0.06985498248641622;
          } else {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)80.00000000000001421) ) ) {
              result[0] += 0.0689431486366741;
            } else {
              result[0] += -0.017362113849393568;
            }
          }
        } else {
          if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)7.500000000000000888) ) ) {
            if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.03342835742364865;
            } else {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)40.00000000000000711) ) ) {
                result[0] += -0.10305534248410175;
              } else {
                if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)7.000000000000000888) ) ) {
                  result[0] += 0.021637750457430063;
                } else {
                  result[0] += -0.05131752733480354;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)10.50000000000000178) ) ) {
              result[0] += 0.02284326651617681;
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.694163918495178667) ) ) {
                result[0] += 0.05632298921525436;
              } else {
                result[0] += -0.0852299451292156;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)15.50000000000000178) ) ) {
          result[0] += 0.09664188350859476;
        } else {
          result[0] += -0.10803721460374056;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
      if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)47227863040.00000763) ) ) {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)56.00000000000000711) ) ) {
            result[0] += -0.10026549972821663;
          } else {
            if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
              result[0] += 0.0003601403714970319;
            } else {
              result[0] += -0.09019510532237851;
            }
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.778982400894165927) ) ) {
            result[0] += -0.07635573317280442;
          } else {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)80.00000000000001421) ) ) {
              if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)29.50000000000000355) ) ) {
                result[0] += -0.051914136159433;
              } else {
                result[0] += 0.04227375384861912;
              }
            } else {
              result[0] += 0.05387534296704808;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.481347560882569248) ) ) {
          result[0] += -0.010922628937012208;
        } else {
          result[0] += -0.10354947673263828;
        }
      }
    } else {
      if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)7.500000000000000888) ) ) {
          if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)104.0000000000000142) ) ) {
              if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)112.0000000000000142) ) ) {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.05576130879760291;
                  } else {
                    result[0] += 0.01785123094335557;
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.284418344497681552) ) ) {
                    result[0] += -0.011252977525823632;
                  } else {
                    result[0] += -0.07246411848720098;
                  }
                }
              } else {
                if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)5.000000000000000888) ) ) {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.48918962478637873) ) ) {
                        if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.834949493408204901) ) ) {
                            result[0] += 0.055710602976835014;
                          } else {
                            result[0] += -0.024732936488225745;
                          }
                        } else {
                          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                            result[0] += -0.03126322317969477;
                          } else {
                            result[0] += 0.06797392517901109;
                          }
                        }
                      } else {
                        result[0] += -0.09030770414205092;
                      }
                    } else {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.302512168884278232) ) ) {
                        result[0] += 0.04945960659230694;
                      } else {
                        if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += -0.06484016713946276;
                        } else {
                          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.868834793567657693) ) ) {
                            result[0] += -0.10026322923591675;
                          } else {
                            result[0] += 0.03860309976600014;
                          }
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                        result[0] += 0.031380172997743865;
                      } else {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.284418344497681552) ) ) {
                          result[0] += 0.05382148600965142;
                        } else {
                          result[0] += -0.07859552038846024;
                        }
                      }
                    } else {
                      result[0] += -0.09560001111145522;
                    }
                  }
                } else {
                  result[0] += 0.05841496035513523;
                }
              }
            } else {
              if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += -0.10035104311014283;
              } else {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += -0.054357208301163876;
                } else {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.917405366897583452) ) ) {
                    result[0] += -0.03811196765406188;
                  } else {
                    result[0] += 0.04207877381968223;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
              result[0] += -0.08871519728941428;
            } else {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += -0.014031962002595858;
              } else {
                result[0] += 0.048504765892923046;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.484580039978028232) ) ) {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += -0.06500003299350259;
            } else {
              if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)5.500000000000000888) ) ) {
                result[0] += 0.02995728093644925;
              } else {
                result[0] += 0.150181215087909;
              }
            }
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.605039834976196733) ) ) {
              result[0] += 0.0005411562519805899;
            } else {
              result[0] += -0.09518876258768094;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.778982400894165927) ) ) {
            result[0] += -0.051790841521015256;
          } else {
            if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)16.50000000000000355) ) ) {
              result[0] += 0.048857874163581436;
            } else {
              result[0] += -0.09961246505012306;
            }
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += -0.07475368830287304;
            } else {
              result[0] += 0.07406352409301846;
            }
          } else {
            result[0] += 0.08888009177662949;
          }
        }
      }
    }
  }
  if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
          result[0] += 0.09796685222123008;
        } else {
          result[0] += 0.04050965810706687;
        }
      } else {
        result[0] += 0.10099581049977771;
      }
    } else {
      if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)17.50000000000000355) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.284418344497681552) ) ) {
          if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.07193347633941377;
            } else {
              result[0] += 0.0622038511283699;
            }
          } else {
            result[0] += 0.016336880090955627;
          }
        } else {
          if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)7.500000000000000888) ) ) {
            if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)26.50000000000000355) ) ) {
              if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.500000000000000444) ) ) {
                if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                  if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)6.500000000000000888) ) ) {
                    result[0] += -0.08613805219956067;
                  } else {
                    result[0] += 0.008983404148528204;
                  }
                } else {
                  result[0] += 0.020484328075895057;
                }
              } else {
                result[0] += 0.021782093145273577;
              }
            } else {
              result[0] += 0.081793661906778;
            }
          } else {
            if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)2.500000000000000444) ) ) {
              result[0] += -0.06824039140640846;
            } else {
              result[0] += 0.07637205659402438;
            }
          }
        }
      } else {
        result[0] += 0.0881801450503539;
      }
    }
  } else {
    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
          if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)7.500000000000000888) ) ) {
            result[0] += -0.09689913767001794;
          } else {
            if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
              result[0] += -0.004700248512154705;
            } else {
              result[0] += -0.09388788240494161;
            }
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.778982400894165927) ) ) {
            result[0] += -0.07327443680263712;
          } else {
            if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)5.500000000000000888) ) ) {
              if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)29.50000000000000355) ) ) {
                result[0] += -0.05295526139882929;
              } else {
                result[0] += 0.04395255951049949;
              }
            } else {
              result[0] += 0.048953178703592594;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.481347560882569248) ) ) {
          result[0] += -0.008656609891228819;
        } else {
          result[0] += -0.10174162962722644;
        }
      }
    } else {
      if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)7.500000000000000888) ) ) {
          if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)104.0000000000000142) ) ) {
              if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)112.0000000000000142) ) ) {
                if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)5.500000000000000888) ) ) {
                  if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
                    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.142630577087403232) ) ) {
                        result[0] += 0.020395643275573274;
                      } else {
                        if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)2.500000000000000444) ) ) {
                            result[0] += 0.056438082863049;
                          } else {
                            result[0] += -0.0388688961692962;
                          }
                        } else {
                          result[0] += -0.10734971221001532;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.736135363578796831) ) ) {
                        result[0] += -0.02843903129904036;
                      } else {
                        result[0] += -0.08861535139202009;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                      result[0] += -0.04094179921423102;
                    } else {
                      result[0] += 0.029691627816743327;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)8.500000000000001776) ) ) {
                    result[0] += 0.03939427604816635;
                  } else {
                    if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)9.500000000000001776) ) ) {
                      result[0] += -0.10921020251903;
                    } else {
                      result[0] += 0.004940819056307815;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.302512168884278232) ) ) {
                      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.357691764831543413) ) ) {
                            result[0] += -0.013449893965784757;
                          } else {
                            result[0] += 0.06097418667465435;
                          }
                        } else {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.04969787597656428) ) ) {
                            result[0] += 0.044985458744446954;
                          } else {
                            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                              result[0] += -0.056505302239005406;
                            } else {
                              result[0] += 0.009014661154722232;
                            }
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.736135363578796831) ) ) {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.978769779205324042) ) ) {
                            result[0] += -0.003577583253351925;
                          } else {
                            result[0] += 0.09401775156333003;
                          }
                        } else {
                          result[0] += 0.027423454726134156;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                        result[0] += 0.040182656634402285;
                      } else {
                        if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += -0.07855477106864354;
                        } else {
                          if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.500000000000000444) ) ) {
                            result[0] += 0.04860916269698112;
                          } else {
                            result[0] += -0.04486991678473756;
                          }
                        }
                      }
                    }
                  } else {
                    result[0] += -0.03134560546761287;
                  }
                } else {
                  result[0] += 0.05277221434341705;
                }
              }
            } else {
              if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += -0.09692963810240235;
              } else {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += -0.05105140403720359;
                } else {
                  result[0] += 0.012328733187069035;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)2.500000000000000444) ) ) {
              result[0] += -0.09063406231152027;
            } else {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += -0.04403970567158931;
              } else {
                result[0] += 0.026667710295501808;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.484580039978028232) ) ) {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += -0.06022471030739762;
            } else {
              if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)11.50000000000000178) ) ) {
                result[0] += 0.1044898754250226;
              } else {
                result[0] += -0.013069743635335869;
              }
            }
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.605039834976196733) ) ) {
              result[0] += 0.0021085441385791637;
            } else {
              result[0] += -0.09336713108879417;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.484580039978028232) ) ) {
            result[0] += -0.09331058906787189;
          } else {
            result[0] += 0.03754092777955998;
          }
        } else {
          if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)17.50000000000000355) ) ) {
            result[0] += 0.0378179309775352;
          } else {
            result[0] += 0.08831376811133256;
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.497866153717041238) ) ) {
            result[0] += 0.0312667423170065;
          } else {
            result[0] += 0.07804441644488945;
          }
        } else {
          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += 0.07305749723887474;
          } else {
            result[0] += 0.005209069350391894;
          }
        }
      } else {
        result[0] += 0.09859235929662685;
      }
    } else {
      if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.778982400894165927) ) ) {
          if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
              if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)13.50000000000000178) ) ) {
                result[0] += -0.01643881576061518;
              } else {
                result[0] += 0.1008079020790514;
              }
            } else {
              result[0] += 0.06361970327336282;
            }
          } else {
            if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
              result[0] += -0.06362236735517664;
            } else {
              result[0] += 0.028829106116659814;
            }
          }
        } else {
          if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)7.500000000000000888) ) ) {
            if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)26.50000000000000355) ) ) {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)56.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.651049375534058505) ) ) {
                  result[0] += 0.029255770431094133;
                } else {
                  result[0] += -0.05311278818803872;
                }
              } else {
                result[0] += 0.017808928014942684;
              }
            } else {
              result[0] += 0.0783194029967873;
            }
          } else {
            result[0] += -0.06745741039788235;
          }
        }
      } else {
        if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)16.50000000000000355) ) ) {
          result[0] += 0.09400077804590966;
        } else {
          result[0] += -0.11208480014794607;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
          if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)5.000000000000000888) ) ) {
            result[0] += -0.09274266617630265;
          } else {
            if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.1822080612182635) ) ) {
                  result[0] += 0.04344563478763405;
                } else {
                  result[0] += -0.13260868166080805;
                }
              } else {
                result[0] += 0.05158530146199263;
              }
            } else {
              result[0] += -0.09239361319124885;
            }
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.778982400894165927) ) ) {
            result[0] += -0.07089827189960951;
          } else {
            if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)10.50000000000000178) ) ) {
              result[0] += 0.024970556249802826;
            } else {
              result[0] += -0.07893951213606032;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.040716171264650214) ) ) {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.516936540603638583) ) ) {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
              result[0] += -0.11098605895735601;
            } else {
              result[0] += -0.002755705586524543;
            }
          } else {
            result[0] += 0.02168774852844363;
          }
        } else {
          result[0] += -0.10060776758395558;
        }
      }
    } else {
      if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)7.500000000000000888) ) ) {
          if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.20590913295745894) ) ) {
              result[0] += -0.11168987325987392;
            } else {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)104.0000000000000142) ) ) {
                if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)112.0000000000000142) ) ) {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)20.00000000000000355) ) ) {
                    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2202.000000000000455) ) ) {
                      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                        result[0] += -0.08543121437767744;
                      } else {
                        result[0] += 0.0014694679133703517;
                      }
                    } else {
                      result[0] += 0.016307271867373373;
                    }
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
                      result[0] += -0.01074665966609068;
                    } else {
                      if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                        result[0] += -0.10055726761653375;
                      } else {
                        result[0] += -0.026070761085372318;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)7.000000000000000888) ) ) {
                      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.48918962478637873) ) ) {
                          if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                            result[0] += -0.014762689121357937;
                          } else {
                            result[0] += 0.027587921191875676;
                          }
                        } else {
                          result[0] += -0.07764703898305608;
                        }
                      } else {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.736135363578796831) ) ) {
                          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
                            result[0] += 0.0007440471608387727;
                          } else {
                            result[0] += 0.06364750726243017;
                          }
                        } else {
                          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                            result[0] += 0.06430388545780352;
                          } else {
                            result[0] += -0.004516514873303448;
                          }
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                        result[0] += 0.003282486855673516;
                      } else {
                        result[0] += -0.10437062579911333;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.232423543930054599) ) ) {
                      result[0] += -0.01880076931252474;
                    } else {
                      result[0] += 0.05750644057435437;
                    }
                  }
                }
              } else {
                result[0] += -0.02226567624944438;
              }
            }
          } else {
            if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += -0.10121314906066839;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.215408444404602495) ) ) {
                  result[0] += 0.047653399316712364;
                } else {
                  result[0] += -0.10553557545333424;
                }
              }
            } else {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += -0.040391210366757295;
              } else {
                result[0] += 0.022382218673635667;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.778982400894165927) ) ) {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += -0.05989615171791586;
            } else {
              if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)11.50000000000000178) ) ) {
                result[0] += 0.09322216959670653;
              } else {
                result[0] += -0.01792953836685458;
              }
            }
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.605039834976196733) ) ) {
              result[0] += 0.0026603355081978716;
            } else {
              result[0] += -0.09131938888497304;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)16.50000000000000355) ) ) {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.605039834976196733) ) ) {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += -0.06710520292602407;
            } else {
              result[0] += 0.06961001556088459;
            }
          } else {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += 0.0415170902251862;
            } else {
              result[0] += 0.08414163412577703;
            }
          }
        } else {
          result[0] += -0.08929013812264686;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
          result[0] += 0.09106205627703667;
        } else {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.994492053985595925) ) ) {
              result[0] += 0.01863398800502658;
            } else {
              result[0] += 0.07741899176711774;
            }
          } else {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.06927420733969314;
            } else {
              if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
                result[0] += -0.01436189545697248;
              } else {
                result[0] += 0.061209898375246964;
              }
            }
          }
        }
      } else {
        result[0] += 0.09622060435825504;
      }
    } else {
      if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)17.50000000000000355) ) ) {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.484580039978028232) ) ) {
          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)112.0000000000000142) ) ) {
              if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)4.500000000000000888) ) ) {
                if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)2.500000000000000444) ) ) {
                  result[0] += -0.025880903710699545;
                } else {
                  result[0] += 0.047091214521824985;
                }
              } else {
                result[0] += 0.09073842153504728;
              }
            } else {
              result[0] += 0.060641799810567026;
            }
          } else {
            result[0] += -0.035993358673196085;
          }
        } else {
          if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)7.500000000000000888) ) ) {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)20.00000000000000355) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.671854496002199042) ) ) {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += -0.06047914598398801;
                } else {
                  result[0] += 0.07140850527072608;
                }
              } else {
                result[0] += 0.03341355576148821;
              }
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.05843427300111318;
                } else {
                  result[0] += 0.0271766378403232;
                }
              } else {
                result[0] += -0.05701067982463082;
              }
            }
          } else {
            if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)2.500000000000000444) ) ) {
              result[0] += -0.06391741520199794;
            } else {
              result[0] += 0.071563242660339;
            }
          }
        }
      } else {
        result[0] += 0.08286784092856295;
      }
    }
  } else {
    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
            result[0] += -0.08459599055917966;
          } else {
            result[0] += -0.018605303759040483;
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.778982400894165927) ) ) {
            result[0] += -0.06972404191203098;
          } else {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)80.00000000000001421) ) ) {
              if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)13.50000000000000178) ) ) {
                result[0] += -0.044705436940268226;
              } else {
                result[0] += 0.036349022434255245;
              }
            } else {
              result[0] += 0.042269947337200055;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.040716171264650214) ) ) {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.516936540603638583) ) ) {
            result[0] += -0.061116288782180256;
          } else {
            result[0] += 0.0187699359507795;
          }
        } else {
          result[0] += -0.09852935579667246;
        }
      }
    } else {
      if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)7.500000000000000888) ) ) {
          if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.100254535675049272) ) ) {
              result[0] += -0.11106340502234054;
            } else {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)104.0000000000000142) ) ) {
                if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)112.0000000000000142) ) ) {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)20.00000000000000355) ) ) {
                    result[0] += 0.009716580853774514;
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
                      result[0] += -0.01255892685888235;
                    } else {
                      if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                        result[0] += -0.09798884635672334;
                      } else {
                        result[0] += -0.025191249799266065;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)7.000000000000000888) ) ) {
                      if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += -0.0009554251415587157;
                      } else {
                        if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                          if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                            result[0] += 0.07110324351068642;
                          } else {
                            result[0] += -0.007493926213514733;
                          }
                        } else {
                          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.736135363578796831) ) ) {
                            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.599987030029298651) ) ) {
                                result[0] += 0.07915209273253049;
                              } else {
                                result[0] += -0.021448763538485474;
                              }
                            } else {
                              result[0] += 0.055475532083730905;
                            }
                          } else {
                            result[0] += -0.016364378409792824;
                          }
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                        result[0] += 0.001879653002304837;
                      } else {
                        result[0] += -0.10118871543036596;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.232423543930054599) ) ) {
                      result[0] += -0.025624011600177177;
                    } else {
                      result[0] += 0.05273559193778276;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += -0.08910808654123846;
                } else {
                  result[0] += -0.007260815584298152;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
              result[0] += -0.08029318197547301;
            } else {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += -0.02267261148679564;
              } else {
                result[0] += 0.03501280822919841;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.778982400894165927) ) ) {
            if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)5.500000000000000888) ) ) {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += -0.05612485109392458;
              } else {
                if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.10498096604734586;
                } else {
                  result[0] += -0.070386890571435;
                }
              }
            } else {
              if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)11.50000000000000178) ) ) {
                result[0] += 0.1275677771013972;
              } else {
                result[0] += -0.028366284387792085;
              }
            }
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.605039834976196733) ) ) {
              result[0] += 0.0007121589299729958;
            } else {
              result[0] += -0.08910023385671491;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.329314231872559482) ) ) {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.718933820724488193) ) ) {
              result[0] += -0.048434238454692746;
            } else {
              result[0] += 0.045695679464349076;
            }
          } else {
            result[0] += 0.056816448722865065;
          }
        } else {
          if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)15.50000000000000178) ) ) {
            result[0] += 0.07691091395960377;
          } else {
            result[0] += -0.07210435608375133;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
            result[0] += 0.08718935527977925;
          } else {
            result[0] += 0.03266697587966139;
          }
        } else {
          result[0] += 0.0933718297686086;
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.142630577087403232) ) ) {
          if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)5.500000000000000888) ) ) {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += -0.01375127283666359;
            } else {
              if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.05950803104136846;
              } else {
                result[0] += 0.012026748916255789;
              }
            }
          } else {
            result[0] += 0.12023083403072594;
          }
        } else {
          if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)7.500000000000000888) ) ) {
            if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)29.50000000000000355) ) ) {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.993164777755738193) ) ) {
                result[0] += 0.012437798962228934;
              } else {
                result[0] += -0.029908535689116018;
              }
            } else {
              result[0] += 0.07270286411855963;
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.694163918495178667) ) ) {
              result[0] += 0.047639940165062344;
            } else {
              result[0] += -0.06963630309539212;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
          if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.20590913295745894) ) ) {
            result[0] += 0.019474288007786746;
          } else {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.051912069320679599) ) ) {
                result[0] += -0.08365994228977874;
              } else {
                if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)25.50000000000000355) ) ) {
                  result[0] += -0.054159102560550426;
                } else {
                  result[0] += 0.05616531724897598;
                }
              }
            } else {
              result[0] += -0.1014057796464175;
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.1822080612182635) ) ) {
            result[0] += 0.0449222382348522;
          } else {
            result[0] += -0.04926795584506215;
          }
        }
      } else {
        if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)11.50000000000000178) ) ) {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.48918962478637873) ) ) {
                if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)17.50000000000000355) ) ) {
                  if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)31.50000000000000355) ) ) {
                    result[0] += 0.02898187351121146;
                  } else {
                    result[0] += 0.09560462460950718;
                  }
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.90474271774292081) ) ) {
                    result[0] += 0.041132658353892695;
                  } else {
                    result[0] += -0.207804991775848;
                  }
                }
              } else {
                result[0] += -0.0602366518032895;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.38936424255371271) ) ) {
                if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( UNLIKELY(  (data[42].missing != -1) && (data[42].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                    result[0] += 0.10119061881018168;
                  } else {
                    if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)112.0000000000000142) ) ) {
                        result[0] += -0.04661524137167472;
                      } else {
                        result[0] += 0.02429249674595274;
                      }
                    } else {
                      result[0] += -0.055161834065798915;
                    }
                  }
                } else {
                  result[0] += 0.09541820978836546;
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.778982400894165927) ) ) {
                  if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.595119953155518466) ) ) {
                    result[0] += -0.06367477360758532;
                  } else {
                    result[0] += -0.11640755302875966;
                  }
                } else {
                  if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)7.500000000000000888) ) ) {
                    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                      if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += 0.0016848998939946386;
                      } else {
                        if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                          result[0] += 0.09345553917700218;
                        } else {
                          result[0] += -0.06988998636776654;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
                        result[0] += -0.07833999538254226;
                      } else {
                        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                            result[0] += -0.06559148018917939;
                          } else {
                            if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.802696108818054643) ) ) {
                              result[0] += -0.019808610064071874;
                            } else {
                              result[0] += 0.07094515656472114;
                            }
                          }
                        } else {
                          result[0] += 0.09464761282007972;
                        }
                      }
                    }
                  } else {
                    result[0] += -0.0929498809428586;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.778982400894165927) ) ) {
              if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)4.500000000000000888) ) ) {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.56849193572998225) ) ) {
                    result[0] += 0.02735210914814252;
                  } else {
                    result[0] += -0.025227276524608763;
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.978769779205324042) ) ) {
                    result[0] += -0.02443156206824032;
                  } else {
                    result[0] += 0.051862219713382764;
                  }
                }
              } else {
                result[0] += 0.10892350891704526;
              }
            } else {
              if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)3072.000000000000455) ) ) {
                  result[0] += 0.04237621731296741;
                } else {
                  result[0] += -0.0914604222218825;
                }
              } else {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                  if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.500000000000000444) ) ) {
                      result[0] += -0.01877074242635328;
                    } else {
                      result[0] += 0.03638849055677677;
                    }
                  } else {
                    result[0] += 0.07330306262079413;
                  }
                } else {
                  if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.154959201812744585) ) ) {
                      result[0] += 0.07805038385203877;
                    } else {
                      result[0] += -0.09398295252275582;
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.978769779205324042) ) ) {
                      result[0] += -0.09721473859768091;
                    } else {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.208071470260621005) ) ) {
                        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                          result[0] += 0.03813840539262431;
                        } else {
                          result[0] += -0.008770789981632929;
                        }
                      } else {
                        result[0] += -0.03356463524608128;
                      }
                    }
                  }
                }
              }
            }
          }
        } else {
          result[0] += -0.07910688205455207;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)16.50000000000000355) ) ) {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.154959201812744585) ) ) {
        result[0] += -0.05095858675641393;
      } else {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += 0.09723668580725431;
        } else {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
            result[0] += -0.10081607593092701;
          } else {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += 0.03229076224498726;
            } else {
              result[0] += 0.07469596242295592;
            }
          }
        }
      }
    } else {
      result[0] += -0.09621680488866075;
    }
  }
  if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)17.50000000000000355) ) ) {
    if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.497866153717041238) ) ) {
            result[0] += 0.026089626928201538;
          } else {
            result[0] += 0.06724197281110146;
          }
        } else {
          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += 0.06371834677485996;
          } else {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += -0.007412831588534176;
            } else {
              result[0] += 0.0510343725287035;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
          result[0] += 0.09706033683886302;
        } else {
          result[0] += 0.020161850379162266;
        }
      }
    } else {
      if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)7.500000000000000888) ) ) {
          if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
            if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.888826131820679155) ) ) {
                result[0] += -0.0190599267263762;
              } else {
                result[0] += -0.10801664616747685;
              }
            } else {
              if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)29.50000000000000355) ) ) {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                    if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)7.000000000000000888) ) ) {
                      if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                        if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)6.500000000000000888) ) ) {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.1822080612182635) ) ) {
                            result[0] += 0.011610198498045109;
                          } else {
                            result[0] += -0.028221900896817934;
                          }
                        } else {
                          if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)8.500000000000001776) ) ) {
                            if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                              result[0] += 0.0624112083598016;
                            } else {
                              result[0] += 0.00529017325764468;
                            }
                          } else {
                            if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)9.500000000000001776) ) ) {
                              result[0] += -0.08977407696696614;
                            } else {
                              if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)17.50000000000000355) ) ) {
                                result[0] += 0.021160684743725393;
                              } else {
                                if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                                  result[0] += -0.10180064296888974;
                                } else {
                                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.384830474853516513) ) ) {
                                    result[0] += 0.015057579230254506;
                                  } else {
                                    result[0] += -0.07278286499490018;
                                  }
                                }
                              }
                            }
                          }
                        }
                      } else {
                        if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)7.000000000000000888) ) ) {
                          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                            result[0] += -0.07749150027252558;
                          } else {
                            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.827801465988160068) ) ) {
                              result[0] += 0.007296022987559632;
                            } else {
                              result[0] += 0.0709351307569651;
                            }
                          }
                        } else {
                          result[0] += -0.12543964267006877;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.90474271774292081) ) ) {
                        result[0] += -0.07568099159380522;
                      } else {
                        result[0] += 0.12214935248192263;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.1822080612182635) ) ) {
                      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)88.00000000000001421) ) ) {
                        if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += -0.04698226382292058;
                        } else {
                          if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                            result[0] += -0.03925228228874684;
                          } else {
                            if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
                              if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                                result[0] += 0.050360939339528746;
                              } else {
                                result[0] += -0.03526830163117372;
                              }
                            } else {
                              result[0] += 0.05228947654693622;
                            }
                          }
                        }
                      } else {
                        result[0] += -0.058331495891442334;
                      }
                    } else {
                      if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                        if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                          result[0] += -0.09550288417050276;
                        } else {
                          result[0] += -0.04824832858152764;
                        }
                      } else {
                        result[0] += 0.006410582400010363;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.029068946838379794) ) ) {
                      if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.500000000000000444) ) ) {
                        result[0] += -0.03332619352736807;
                      } else {
                        result[0] += 0.03826356273624106;
                      }
                    } else {
                      result[0] += -0.06723475474248562;
                    }
                  } else {
                    if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += -0.0716288883344946;
                    } else {
                      if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
                          result[0] += 0.0018058456638326695;
                        } else {
                          result[0] += -0.10141812181396725;
                        }
                      } else {
                        if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += -0.06841654961591918;
                        } else {
                          result[0] += 0.024039990305950054;
                        }
                      }
                    }
                  }
                }
              } else {
                result[0] += 0.04216437841169557;
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.39281225204467951) ) ) {
              if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += -0.10147573495699128;
              } else {
                result[0] += -0.00027434153770671015;
              }
            } else {
              result[0] += -0.08225271593377936;
            }
          }
        } else {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.827801465988160068) ) ) {
            result[0] += -0.006540990268669782;
          } else {
            result[0] += -0.08582298018307982;
          }
        }
      } else {
        if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)5.500000000000000888) ) ) {
          if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.694163918495178667) ) ) {
              result[0] += -0.06775402477300095;
            } else {
              result[0] += -0.10038178201651518;
            }
          } else {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
              result[0] += 0.010443497332545223;
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.02604460716247603) ) ) {
                result[0] += -0.020888065480502975;
              } else {
                result[0] += -0.07483603118011954;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.778982400894165927) ) ) {
            if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)11.50000000000000178) ) ) {
              result[0] += 0.08297054527983144;
            } else {
              result[0] += -0.037146791875249746;
            }
          } else {
            if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.025836731264668567;
            } else {
              if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                result[0] += -0.08921031752013554;
              } else {
                result[0] += -0.017867606741884312;
              }
            }
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)16.50000000000000355) ) ) {
      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
        result[0] += 0.02393173933675446;
      } else {
        if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)2.500000000000000444) ) ) {
          result[0] += 0.09459463435079278;
        } else {
          result[0] += 0.05119538529878292;
        }
      }
    } else {
      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.88435244560241788) ) ) {
        if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
          result[0] += 0.0349433410722659;
        } else {
          result[0] += -0.09323810976226121;
        }
      } else {
        result[0] += -0.09023649247590795;
      }
    }
  }
  if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)292040998912.000061) ) ) {
          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
            result[0] += 0.08198441382032928;
          } else {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.994492053985595925) ) ) {
                result[0] += 0.016282971106103057;
              } else {
                result[0] += 0.06872175514487852;
              }
            } else {
              result[0] += 0.010162028979634346;
            }
          }
        } else {
          result[0] += 0.08856625522613178;
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.142630577087403232) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.65882921218872248) ) ) {
            result[0] += 0.015795228784473734;
          } else {
            result[0] += 0.05373948486204752;
          }
        } else {
          if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)7.500000000000000888) ) ) {
            if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.018881309514170233;
            } else {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)40.00000000000000711) ) ) {
                result[0] += -0.10422876729881343;
              } else {
                if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)7.000000000000000888) ) ) {
                  result[0] += 0.005025983266791208;
                } else {
                  result[0] += -0.06306425302526693;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.605039834976196733) ) ) {
              result[0] += 0.044429497972084044;
            } else {
              result[0] += -0.06503123892674119;
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)16.50000000000000355) ) ) {
        if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.595119953155518466) ) ) {
          result[0] += 0.09878428672445276;
        } else {
          result[0] += -0.054314428325705204;
        }
      } else {
        result[0] += -0.102571484908857;
      }
    }
  } else {
    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
        if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.20590913295745894) ) ) {
          result[0] += 0.0187748986837988;
        } else {
          if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.088880300521851474) ) ) {
              result[0] += -0.08081549762915112;
            } else {
              if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)25.50000000000000355) ) ) {
                result[0] += -0.04950968426014632;
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.070054531097412998) ) ) {
                  result[0] += 0.06879803802732075;
                } else {
                  result[0] += -0.19777213170243257;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)2.65906000137329146) ) ) {
              result[0] += -0.1006782554868459;
            } else {
              result[0] += 0.6968961268623395;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.1822080612182635) ) ) {
          result[0] += 0.03755592041159002;
        } else {
          result[0] += -0.04800516483934794;
        }
      }
    } else {
      if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)11.50000000000000178) ) ) {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.067782521247864214) ) ) {
                result[0] += -0.06588613535605987;
              } else {
                if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.10290670394897639) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.357691764831543413) ) ) {
                      result[0] += 0.10555731861192524;
                    } else {
                      result[0] += -0.10830124852841241;
                    }
                  } else {
                    result[0] += -0.09278172839804397;
                  }
                } else {
                  result[0] += 0.029093199712004747;
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.38936424255371271) ) ) {
                if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += 0.09289796868249983;
                  } else {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)120.0000000000000142) ) ) {
                      if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)112.0000000000000142) ) ) {
                        result[0] += -0.03756503515493362;
                      } else {
                        result[0] += 0.018507144892992315;
                      }
                    } else {
                      result[0] += -0.09122956853744597;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                    result[0] += 0.1063482731491793;
                  } else {
                    result[0] += -0.01581307476791559;
                  }
                }
              } else {
                if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)112.0000000000000142) ) ) {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                    result[0] += -0.010038621753315498;
                  } else {
                    if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)2.500000000000000444) ) ) {
                      result[0] += -0.09347792311862048;
                    } else {
                      if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.44140100479126021) ) ) {
                        result[0] += -0.05794454581377694;
                      } else {
                        result[0] += 0.18067499173759227;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.029068946838379794) ) ) {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += -0.07279623477070717;
                    } else {
                      result[0] += -0.016548669379785576;
                    }
                  } else {
                    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                      result[0] += 0.058686319171915836;
                    } else {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.397998809814454013) ) ) {
                        result[0] += -0.0736218110108273;
                      } else {
                        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)20.00000000000000355) ) ) {
                          result[0] += 0.05764182937604393;
                        } else {
                          result[0] += -0.05212679939331308;
                        }
                      }
                    }
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.516936540603638583) ) ) {
              if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)5.500000000000000888) ) ) {
                if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.0101229172695735;
                } else {
                  result[0] += 0.036316952651209614;
                }
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.379217386245728427) ) ) {
                  result[0] += 0.10075208734843613;
                } else {
                  result[0] += -0.030673392463589494;
                }
              }
            } else {
              if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                  result[0] += 0.033304146665390714;
                } else {
                  result[0] += -0.09768432642908104;
                }
              } else {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                  result[0] += 0.029402057894850443;
                } else {
                  if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.09016480848688807;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.834949493408204901) ) ) {
                      result[0] += -0.10883653631070443;
                    } else {
                      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                        result[0] += 0.019657967640469915;
                      } else {
                        result[0] += -0.027035579654393606;
                      }
                    }
                  }
                }
              }
            }
          }
        } else {
          result[0] += -0.07359575770407382;
        }
      } else {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.48738741874694913) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.172047138214112216) ) ) {
              result[0] += -0.07137330737735972;
            } else {
              result[0] += 0.016350688581463067;
            }
          } else {
            result[0] += 0.04775819833805149;
          }
        } else {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.48918962478637873) ) ) {
            result[0] += 0.0692752178714958;
          } else {
            result[0] += -0.08385037849079242;
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
          result[0] += 0.02982186226230777;
        } else {
          result[0] += 0.08524931872464916;
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
          if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)5.500000000000000888) ) ) {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += 0.010880881237966876;
            } else {
              if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.05472459523748498;
              } else {
                result[0] += -0.012718580522976211;
              }
            }
          } else {
            result[0] += 0.1161791609969882;
          }
        } else {
          if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)26.50000000000000355) ) ) {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.48738741874694913) ) ) {
                result[0] += 0.008690688899658463;
              } else {
                result[0] += -0.05499840414497599;
              }
            } else {
              result[0] += 0.05762907597481927;
            }
          } else {
            result[0] += -0.03972763918161983;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5.830332040786744052) ) ) {
        if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)16.50000000000000355) ) ) {
          result[0] += 0.09209555454526484;
        } else {
          result[0] += -0.09861887743819048;
        }
      } else {
        result[0] += -0.09688932959879264;
      }
    }
  } else {
    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)56.00000000000000711) ) ) {
            result[0] += -0.0861827610547203;
          } else {
            if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
              result[0] += 0.0007863095087873246;
            } else {
              result[0] += -0.0801036848032219;
            }
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.778982400894165927) ) ) {
            result[0] += -0.06828702258276668;
          } else {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)120.0000000000000142) ) ) {
              result[0] += 0.005225272757635662;
            } else {
              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += 0.02036109126056002;
              } else {
                result[0] += 0.11278957460839617;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.516936540603638583) ) ) {
            result[0] += -0.07377143848843962;
          } else {
            result[0] += 0.01575278473891832;
          }
        } else {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
            result[0] += -0.10125998316439136;
          } else {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += -0.09565428378414176;
            } else {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                result[0] += 0.054138978820136935;
              } else {
                result[0] += -0.0602622915934569;
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)17.50000000000000355) ) ) {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.208071470260621005) ) ) {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.778982400894165927) ) ) {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.962127923965454546) ) ) {
                  result[0] += -0.049708851842597594;
                } else {
                  result[0] += 0.03529342496762252;
                }
              } else {
                if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.284418344497681552) ) ) {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.134366035461426669) ) ) {
                    result[0] += 0.0013775434146908643;
                  } else {
                    result[0] += -0.06336422479483084;
                  }
                } else {
                  result[0] += -0.10036170372814057;
                }
              }
            } else {
              if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)80.00000000000001421) ) ) {
                  result[0] += 0.009355441168092204;
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.172047138214112216) ) ) {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
                      result[0] += -0.09333939648205339;
                    } else {
                      result[0] += 0.06350617509984101;
                    }
                  } else {
                    result[0] += 0.06346761011586101;
                  }
                }
              } else {
                result[0] += -0.0638337730343163;
              }
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.516936540603638583) ) ) {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += 0.01433722215118023;
              } else {
                if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)5.500000000000000888) ) ) {
                  if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                    result[0] += -0.015237641549475206;
                  } else {
                    result[0] += 0.039772202796116686;
                  }
                } else {
                  result[0] += 0.0993612518249183;
                }
              }
            } else {
              if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                result[0] += -0.09090502058316188;
              } else {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
                  result[0] += -0.09447914352074277;
                } else {
                  if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)8.500000000000001776) ) ) {
                    if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)20.00000000000000355) ) ) {
                      result[0] += 0.0271002855039171;
                    } else {
                      result[0] += -0.022242492257519224;
                    }
                  } else {
                    result[0] += -0.07315593765108734;
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)10.50000000000000178) ) ) {
            if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)6.500000000000000888) ) ) {
              if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)20.00000000000000355) ) ) {
                  if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)2.500000000000000444) ) ) {
                    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                      result[0] += 0.11573297908525859;
                    } else {
                      result[0] += -0.012158858718168042;
                    }
                  } else {
                    result[0] += -0.07000276132737064;
                  }
                } else {
                  result[0] += -0.07566490941369093;
                }
              } else {
                if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                  result[0] += -0.023556721495817747;
                } else {
                  result[0] += 0.048090396241992156;
                }
              }
            } else {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += 0.06458388683142856;
              } else {
                if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                  result[0] += 0.009921189840867081;
                } else {
                  result[0] += -0.16008297420213427;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)29.50000000000000355) ) ) {
              result[0] += -0.08929674847174507;
            } else {
              result[0] += 0.017994316917998313;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.484580039978028232) ) ) {
            result[0] += -0.09301290843790994;
          } else {
            if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)16.50000000000000355) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.484580039978028232) ) ) {
                result[0] += -0.06735606416258783;
              } else {
                result[0] += 0.04136257252132436;
              }
            } else {
              result[0] += -0.055381275550406554;
            }
          }
        } else {
          if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
            if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += 0.03565163280878804;
            } else {
              result[0] += 0.08508077923326124;
            }
          } else {
            result[0] += 0.03447405828418454;
          }
        }
      }
    }
  }
  if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
            result[0] += 0.0817971877078319;
          } else {
            result[0] += 0.022116186705376437;
          }
        } else {
          result[0] += 0.08320358277275408;
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.778982400894165927) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.79870843887329279) ) ) {
            if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.10005313091614616;
            } else {
              result[0] += 0.02494739003044897;
            }
          } else {
            if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)11.50000000000000178) ) ) {
              result[0] += 0.06492063926876566;
            } else {
              result[0] += -0.11027165258559163;
            }
          }
        } else {
          if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)7.500000000000000888) ) ) {
            if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)29.50000000000000355) ) ) {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)56.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.651049375534058505) ) ) {
                  result[0] += 0.01733647117968024;
                } else {
                  result[0] += -0.05547357871126688;
                }
              } else {
                result[0] += 0.006025961405250236;
              }
            } else {
              result[0] += 0.056994946585175454;
            }
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.605039834976196733) ) ) {
              result[0] += 0.050347676961844993;
            } else {
              result[0] += -0.06879274809123018;
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)16.50000000000000355) ) ) {
        if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.595119953155518466) ) ) {
          result[0] += 0.09621360531144729;
        } else {
          result[0] += -0.056604799809492595;
        }
      } else {
        result[0] += -0.09599483080324472;
      }
    }
  } else {
    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
      if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)56.00000000000000711) ) ) {
            result[0] += -0.0834535854876948;
          } else {
            if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
              result[0] += -0.00211747498734041;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.599987030029298651) ) ) {
                result[0] += 0.24169292143294296;
              } else {
                result[0] += -0.0857072314491441;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.778982400894165927) ) ) {
            result[0] += -0.06514920131036607;
          } else {
            if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)8.500000000000001776) ) ) {
              result[0] += 0.019032459798108477;
            } else {
              result[0] += -0.08043687660152905;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.040716171264650214) ) ) {
          result[0] += -0.010330578088829027;
        } else {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
            result[0] += -0.09988160418015901;
          } else {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += -0.09401284929057993;
            } else {
              if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                result[0] += -0.05964946001350107;
              } else {
                result[0] += 0.04975825231532568;
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)11.50000000000000178) ) ) {
          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)20.00000000000000355) ) ) {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.718933820724488193) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.718933820724488193) ) ) {
                if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)12.50000000000000178) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                      if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.827801465988160068) ) ) {
                        result[0] += -0.01551319540812096;
                      } else {
                        result[0] += -0.07468836286216889;
                      }
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.055836200714113104) ) ) {
                        result[0] += -0.041734479285158416;
                      } else {
                        result[0] += 0.058465383912310445;
                      }
                    }
                  } else {
                    result[0] += 0.03375256236057212;
                  }
                } else {
                  result[0] += 0.06289644013010218;
                }
              } else {
                if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)7.500000000000000888) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.41211462020874201) ) ) {
                    result[0] += -0.1239352037883934;
                  } else {
                    result[0] += 0.009819733144939231;
                  }
                } else {
                  result[0] += -0.07290788196561247;
                }
              }
            } else {
              if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)5.500000000000000888) ) ) {
                if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)6.500000000000000888) ) ) {
                  result[0] += -0.02074226077961671;
                } else {
                  if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)8.500000000000001776) ) ) {
                    result[0] += 0.05677839570036167;
                  } else {
                    result[0] += -0.005346648120863993;
                  }
                }
              } else {
                result[0] += -0.07454097191117696;
              }
            }
          } else {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)56.00000000000000711) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.651049375534058505) ) ) {
                result[0] += -0.02589810442188144;
              } else {
                result[0] += -0.09168051173201466;
              }
            } else {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.302512168884278232) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.7512402534484881) ) ) {
                    result[0] += 0.03914090776179245;
                  } else {
                    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                      if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += -0.010373151414087373;
                      } else {
                        result[0] += 0.05714654668938507;
                      }
                    } else {
                      if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                          result[0] += -0.08820094024128432;
                        } else {
                          result[0] += -0.022620301694513685;
                        }
                      } else {
                        result[0] += 0.022965287040751653;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.06397276722967875;
                  } else {
                    if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)7.000000000000000888) ) ) {
                      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                        if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.500000000000000444) ) ) {
                          result[0] += 0.06379551092688077;
                        } else {
                          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                            result[0] += 0.045031116949012774;
                          } else {
                            result[0] += -0.05032703882290262;
                          }
                        }
                      } else {
                        result[0] += -0.04842459050627386;
                      }
                    } else {
                      result[0] += -0.06319766977178144;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                  result[0] += -0.009801950225669643;
                } else {
                  result[0] += -0.08664458689646397;
                }
              }
            }
          }
        } else {
          result[0] += -0.0634694767705428;
        }
      } else {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.285887241363526279) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.141444921493531162) ) ) {
              result[0] += -0.052427371552007995;
            } else {
              result[0] += 0.04072960131484496;
            }
          } else {
            result[0] += 0.044819143550510865;
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.484580039978028232) ) ) {
            result[0] += -1.132596161752091e-05;
          } else {
            result[0] += 0.06584771427859025;
          }
        }
      }
    }
  }
  if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
            result[0] += 0.07880116962006896;
          } else {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.994492053985595925) ) ) {
                result[0] += 0.009736416775554271;
              } else {
                result[0] += 0.05964190233495133;
              }
            } else {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.04899847274039457;
              } else {
                if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
                  if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.802696108818054643) ) ) {
                    result[0] += -0.010862946882015632;
                  } else {
                    result[0] += -0.10864109076519841;
                  }
                } else {
                  result[0] += 0.05063381848166737;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
            result[0] += 0.08887570596834465;
          } else {
            result[0] += -0.006500948029777621;
          }
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.1822080612182635) ) ) {
            result[0] += 0.0020688063505258617;
          } else {
            if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
              if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)11.50000000000000178) ) ) {
                if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)4.500000000000000888) ) ) {
                  if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.05338318977200751;
                  } else {
                    result[0] += -0.009286536291260567;
                  }
                } else {
                  result[0] += 0.1135609655379679;
                }
              } else {
                result[0] += -0.0697569018737671;
              }
            } else {
              result[0] += -0.05852833103318338;
            }
          }
        } else {
          if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)26.50000000000000355) ) ) {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.48738741874694913) ) ) {
                result[0] += 0.0048318000192525395;
              } else {
                result[0] += -0.050651634932451894;
              }
            } else {
              result[0] += 0.048940802003482775;
            }
          } else {
            result[0] += -0.04405539888646264;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)16.50000000000000355) ) ) {
        result[0] += 0.091566227646334;
      } else {
        result[0] += -0.09165688501748576;
      }
    }
  } else {
    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.516936540603638583) ) ) {
          result[0] += -0.0775391142713405;
        } else {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.516936540603638583) ) ) {
              result[0] += -0.08827303822689919;
            } else {
              result[0] += -0.015294035760635455;
            }
          } else {
            result[0] += 0.02199435416493988;
          }
        }
      } else {
        if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.20590913295745894) ) ) {
          result[0] += 0.016891268431672723;
        } else {
          result[0] += -0.09640567278752832;
        }
      }
    } else {
      if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)11.50000000000000178) ) ) {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.48918962478637873) ) ) {
                result[0] += 0.023689395038323263;
              } else {
                result[0] += -0.08918158463846614;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.671854496002199042) ) ) {
                if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += -0.0016696335607084706;
                } else {
                  result[0] += 0.1115009629365761;
                }
              } else {
                if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)112.0000000000000142) ) ) {
                  result[0] += -0.08061829588662395;
                } else {
                  if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.339395284652710849) ) ) {
                    result[0] += -0.057707440028119755;
                  } else {
                    result[0] += 0.018114986005017313;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.516936540603638583) ) ) {
              if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)5.500000000000000888) ) ) {
                if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)2.500000000000000444) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.736135363578796831) ) ) {
                      if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                          if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.827801465988160068) ) ) {
                            result[0] += 0.01585202123392704;
                          } else {
                            result[0] += -0.04460908705089758;
                          }
                        } else {
                          result[0] += 0.037380503634240565;
                        }
                      } else {
                        result[0] += -0.0621898484448844;
                      }
                    } else {
                      if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.500000000000000444) ) ) {
                        result[0] += -0.07791600719600558;
                      } else {
                        result[0] += -0.0032593056120010784;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)8.500000000000001776) ) ) {
                      result[0] += 0.041498712227781;
                    } else {
                      result[0] += -0.02626201772505094;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)88.00000000000001421) ) ) {
                    if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
                      if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)7.000000000000000888) ) ) {
                        if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)112.0000000000000142) ) ) {
                          result[0] += -0.008699460475219056;
                        } else {
                          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)6.000000000000000888) ) ) {
                            result[0] += -0.021358229474371698;
                          } else {
                            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.30829524993896662) ) ) {
                              result[0] += 0.05284846047087911;
                            } else {
                              result[0] += 0.016220942987884294;
                            }
                          }
                        }
                      } else {
                        result[0] += -0.07494517038737031;
                      }
                    } else {
                      result[0] += 0.039879745354861496;
                    }
                  } else {
                    result[0] += -0.016737539982792614;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.379217386245728427) ) ) {
                  result[0] += 0.0826085322539373;
                } else {
                  result[0] += -0.04137766504530384;
                }
              }
            } else {
              if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)5.500000000000000888) ) ) {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += 0.0035076268777738603;
                } else {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.020127415657043901) ) ) {
                    result[0] += -0.11994235978937609;
                  } else {
                    if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                      result[0] += -0.09880688623980208;
                    } else {
                      if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                        result[0] += -0.03316502772661252;
                      } else {
                        result[0] += 0.036647257782857205;
                      }
                    }
                  }
                }
              } else {
                result[0] += -0.0677224664583298;
              }
            }
          }
        } else {
          result[0] += -0.06771076260532986;
        }
      } else {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
          result[0] += -0.016172228275227057;
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.744781017303467685) ) ) {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.172047138214112216) ) ) {
              result[0] += -0.05885205821569091;
            } else {
              result[0] += 0.046552252390382194;
            }
          } else {
            if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)16.50000000000000355) ) ) {
              result[0] += 0.059186824169314645;
            } else {
              result[0] += -0.057666254258649755;
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
          if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
            result[0] += 0.01022759588029891;
          } else {
            result[0] += 0.03862810072842894;
          }
        } else {
          if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
            result[0] += 0.08654005161253721;
          } else {
            result[0] += 0.01207399016560161;
          }
        }
      } else {
        if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)7.500000000000000888) ) ) {
          if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
            if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.888826131820679155) ) ) {
                result[0] += -0.012518527214982087;
              } else {
                result[0] += -0.10115915241690553;
              }
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)137422176256.0000153) ) ) {
                  if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)7.000000000000000888) ) ) {
                    if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)7.000000000000000888) ) ) {
                      if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
                        if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += -0.0006763491729051825;
                        } else {
                          result[0] += 0.0736432062631751;
                        }
                      } else {
                        if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)6.500000000000000888) ) ) {
                          if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.802696108818054643) ) ) {
                            result[0] += -0.02459444594978384;
                          } else {
                            result[0] += 0.014170236995507183;
                          }
                        } else {
                          if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)4.500000000000000888) ) ) {
                            if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                              result[0] += 0.06415555647559626;
                            } else {
                              result[0] += 0.0035962541514512033;
                            }
                          } else {
                            if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)25.50000000000000355) ) ) {
                              if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)9.500000000000001776) ) ) {
                                result[0] += -0.08832810725474301;
                              } else {
                                if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)8.500000000000001776) ) ) {
                                  result[0] += -0.0012352993263458512;
                                } else {
                                  result[0] += -0.08123268766562979;
                                }
                              }
                            } else {
                              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.90474271774292081) ) ) {
                                result[0] += 0.028766063473651737;
                              } else {
                                if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)10.50000000000000178) ) ) {
                                  result[0] += 0.003195572570570543;
                                } else {
                                  result[0] += -0.2654786363099783;
                                }
                              }
                            }
                          }
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
                        result[0] += -0.13999983080784453;
                      } else {
                        result[0] += -0.018152338116399706;
                      }
                    }
                  } else {
                    result[0] += -0.05866570062984098;
                  }
                } else {
                  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += -0.06470881252767431;
                  } else {
                    if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.059492386873058356;
                    } else {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.445705175399781162) ) ) {
                        if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)2.500000000000000444) ) ) {
                          result[0] += 0.024889786507918944;
                        } else {
                          result[0] += -0.052916041928531436;
                        }
                      } else {
                        result[0] += -0.02826789168603516;
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += -0.04407145954992673;
                } else {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.861792564392090288) ) ) {
                      result[0] += 0.008693525741975008;
                    } else {
                      result[0] += -0.053634564443117906;
                    }
                  } else {
                    if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.004313610257018932;
                    } else {
                      if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += -0.07528471335747328;
                      } else {
                        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)80.00000000000001421) ) ) {
                          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.484580039978028232) ) ) {
                            result[0] += 0.0329540109448191;
                          } else {
                            if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)4.500000000000000888) ) ) {
                              if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                                if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)137422176256.0000153) ) ) {
                                  result[0] += 0.10118818805259691;
                                } else {
                                  result[0] += -0.014426438363851872;
                                }
                              } else {
                                if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
                                  result[0] += -0.019658278272820684;
                                } else {
                                  result[0] += -0.09969834843711418;
                                }
                              }
                            } else {
                              if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.778982400894165927) ) ) {
                                  result[0] += 0.02268165722746833;
                                } else {
                                  result[0] += -0.04889293865885333;
                                }
                              } else {
                                if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                                  result[0] += -0.10370729676604931;
                                } else {
                                  result[0] += 0.05979007781974275;
                                }
                              }
                            }
                          }
                        } else {
                          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.484580039978028232) ) ) {
                            result[0] += -0.056018428457727115;
                          } else {
                            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.302512168884278232) ) ) {
                              if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)3.500000000000000444) ) ) {
                                result[0] += 0.07491239696462776;
                              } else {
                                result[0] += 0.020742883146777778;
                              }
                            } else {
                              result[0] += 0.08980107813179172;
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
            if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.802696108818054643) ) ) {
              result[0] += -0.07606446134093792;
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += 0.0031817534959861227;
              } else {
                result[0] += -0.06506175243154748;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.827801465988160068) ) ) {
            result[0] += 0.0008449815295255878;
          } else {
            result[0] += -0.07589020485314764;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)5.500000000000000888) ) ) {
        if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.694163918495178667) ) ) {
            result[0] += -0.06266185705909492;
          } else {
            result[0] += -0.09733207260436047;
          }
        } else {
          if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.02604460716247603) ) ) {
              result[0] += -0.021361829858782447;
            } else {
              result[0] += -0.07423278946530773;
            }
          } else {
            result[0] += 0.009453662025345491;
          }
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.827801465988160068) ) ) {
          if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)13.50000000000000178) ) ) {
            result[0] += -0.006499952344036704;
          } else {
            result[0] += 0.0783567977284855;
          }
        } else {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.605039834976196733) ) ) {
            result[0] += 0.0395320215548616;
          } else {
            result[0] += -0.06664549595651935;
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)16.50000000000000355) ) ) {
      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.605039834976196733) ) ) {
        result[0] += -0.023266477865942995;
      } else {
        if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)2.500000000000000444) ) ) {
          if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += 0.034959929313824224;
          } else {
            result[0] += 0.08631101613166339;
          }
        } else {
          result[0] += 0.04386701760310352;
        }
      }
    } else {
      result[0] += -0.08273455741286773;
    }
  }
  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
            result[0] += 0.07443738814099887;
          } else {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.994492053985595925) ) ) {
                result[0] += 0.008907854801972717;
              } else {
                result[0] += 0.05481559362671484;
              }
            } else {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                result[0] += -0.09259606633252082;
              } else {
                result[0] += 0.006366217421635234;
              }
            }
          }
        } else {
          result[0] += 0.07636661400561912;
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.142630577087403232) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.65882921218872248) ) ) {
            result[0] += 0.004477315286373901;
          } else {
            if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)3.500000000000000444) ) ) {
              if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
                result[0] += 0.04966019833393325;
              } else {
                result[0] += -0.05293832060475382;
              }
            } else {
              result[0] += -0.011634727469082498;
            }
          }
        } else {
          if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += 0.002151690024412282;
          } else {
            result[0] += -0.04615113065590558;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)16.50000000000000355) ) ) {
        result[0] += 0.08733838278672436;
      } else {
        result[0] += -0.08463253047161849;
      }
    }
  } else {
    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.516936540603638583) ) ) {
          result[0] += -0.07477176910463719;
        } else {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.516936540603638583) ) ) {
              result[0] += -0.0845785040937252;
            } else {
              result[0] += -0.013106845765731241;
            }
          } else {
            result[0] += 0.019250643847817314;
          }
        }
      } else {
        if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.938867926597595659) ) ) {
          result[0] += 0.02839271727741971;
        } else {
          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)2.65906000137329146) ) ) {
            result[0] += -0.09459142819788459;
          } else {
            result[0] += 0.41727386502739255;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)11.50000000000000178) ) ) {
          if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.48918962478637873) ) ) {
                  if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)31.50000000000000355) ) ) {
                      if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)13.50000000000000178) ) ) {
                        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.744781017303467685) ) ) {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.13987779617309748) ) ) {
                            result[0] += 0.06375457825491648;
                          } else {
                            result[0] += -0.02854966403545488;
                          }
                        } else {
                          result[0] += 0.02867967273389345;
                        }
                      } else {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.48738741874694913) ) ) {
                          if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)21.50000000000000355) ) ) {
                            result[0] += -0.043947749516822926;
                          } else {
                            result[0] += 0.06315076857014039;
                          }
                        } else {
                          result[0] += -0.09641187782560443;
                        }
                      }
                    } else {
                      result[0] += 0.06472114454287349;
                    }
                  } else {
                    result[0] += -0.06896180392666348;
                  }
                } else {
                  result[0] += -0.08710470635340506;
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.19876670837402521) ) ) {
                  if ( UNLIKELY(  (data[42].missing != -1) && (data[42].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                    result[0] += 0.08737888823401659;
                  } else {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)120.0000000000000142) ) ) {
                      result[0] += 0.0021518544296502966;
                    } else {
                      result[0] += -0.08565941238875238;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.07754880315465718;
                  } else {
                    result[0] += -0.03691444473125269;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.827801465988160068) ) ) {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.79870843887329279) ) ) {
                    result[0] += 0.012277809697796786;
                  } else {
                    result[0] += -0.0403572566147493;
                  }
                } else {
                  if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.671854496002199042) ) ) {
                        result[0] += -0.08087994872052101;
                      } else {
                        result[0] += 0.015681014317156655;
                      }
                    } else {
                      result[0] += 0.05127146514861828;
                    }
                  } else {
                    result[0] += 0.08518099784110676;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                  if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                      result[0] += 0.04080961582359149;
                    } else {
                      result[0] += -0.083381747330482;
                    }
                  } else {
                    result[0] += 0.024028544497412103;
                  }
                } else {
                  if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.0786707920655065;
                  } else {
                    if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
                      if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)112.0000000000000142) ) ) {
                        result[0] += -0.06030006883285424;
                      } else {
                        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.700598716735840066) ) ) {
                          result[0] += -0.1334254181210136;
                        } else {
                          result[0] += 0.015444157003682386;
                        }
                      }
                    } else {
                      result[0] += -0.07233619542486748;
                    }
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.41211462020874201) ) ) {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)88.00000000000001421) ) ) {
                    result[0] += 0.06222302623284301;
                  } else {
                    result[0] += -0.0016057077689154418;
                  }
                } else {
                  result[0] += -0.06658193380949613;
                }
              } else {
                result[0] += -0.022840588542912516;
              }
            } else {
              if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.05080914235197394;
              } else {
                result[0] += 0.032598386736609114;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += -0.08217104038731132;
          } else {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += -0.06960608529848092;
            } else {
              result[0] += 0.06314955670803026;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.778982400894165927) ) ) {
            result[0] += -0.08042026560812694;
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.827801465988160068) ) ) {
              result[0] += -0.08869901459825409;
            } else {
              result[0] += 0.02197234976221566;
            }
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.484580039978028232) ) ) {
            result[0] += -0.0038164370592601547;
          } else {
            result[0] += 0.05884694945323574;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
            if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
              if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                result[0] += 0.011101940077202558;
              } else {
                result[0] += -0.08805997339859153;
              }
            } else {
              result[0] += 0.03725626688160692;
            }
          } else {
            result[0] += 0.07423815641303205;
          }
        } else {
          if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.64763975143432706) ) ) {
            if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.10375560152050181;
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.232423543930054599) ) ) {
                if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.030733718216402833;
                } else {
                  if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                    result[0] += 0.0074430856094294786;
                  } else {
                    result[0] += -0.08932850595847043;
                  }
                }
              } else {
                if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)26.50000000000000355) ) ) {
                  if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)232.0000000000000284) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.475347042083742011) ) ) {
                        result[0] += -0.12571704850056578;
                      } else {
                        result[0] += -0.030253109618379428;
                      }
                    } else {
                      result[0] += 0.08286329210445015;
                    }
                  } else {
                    if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)56.00000000000000711) ) ) {
                      result[0] += 0.024204804430049246;
                    } else {
                      result[0] += -0.02606385594323697;
                    }
                  }
                } else {
                  result[0] += 0.05099477086799397;
                }
              }
            }
          } else {
            result[0] += 0.08925405342961536;
          }
        }
      } else {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
          if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.938867926597595659) ) ) {
            result[0] += 0.05187809305143451;
          } else {
            if ( LIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.09173656399317388;
            } else {
              if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)88.00000000000001421) ) ) {
                  result[0] += 0.02747817847478959;
                } else {
                  result[0] += -0.06256193855630515;
                }
              } else {
                result[0] += -0.0731098764409725;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
            if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.500000000000000888) ) ) {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                  if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.802696108818054643) ) ) {
                    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.07459576844835267;
                    } else {
                      if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                        if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.500000000000000444) ) ) {
                          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.481121778488159624) ) ) {
                            result[0] += 0.0965527561135925;
                          } else {
                            result[0] += -0.03026529490701695;
                          }
                        } else {
                          result[0] += 0.01649109209075497;
                        }
                      } else {
                        result[0] += 0.0660159848694109;
                      }
                    }
                  } else {
                    result[0] += -0.07863572243877634;
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.85341548919677912) ) ) {
                    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)112.0000000000000142) ) ) {
                      result[0] += -0.05804743192763047;
                    } else {
                      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)120.0000000000000142) ) ) {
                        result[0] += 0.01907239192280933;
                      } else {
                        result[0] += -0.09119169901323144;
                      }
                    }
                  } else {
                    result[0] += -0.07034460191462033;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.484580039978028232) ) ) {
                  if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                    if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)2.500000000000000444) ) ) {
                      result[0] += -0.01924691268307313;
                    } else {
                      result[0] += 0.037545290068705606;
                    }
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.30829524993896662) ) ) {
                      result[0] += 0.05409137482047897;
                    } else {
                      result[0] += 0.011553192238037973;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                      result[0] += 0.037012612642932666;
                    } else {
                      result[0] += -0.08456382248125585;
                    }
                  } else {
                    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                      if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
                        if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.500000000000000444) ) ) {
                          result[0] += -0.02051728550736902;
                        } else {
                          result[0] += 0.046308491711354705;
                        }
                      } else {
                        if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += -0.16925314442295447;
                        } else {
                          result[0] += 0.10257253968508095;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += -0.07943160143302665;
                      } else {
                        result[0] += -0.0110148811888495;
                      }
                    }
                  }
                }
              }
            } else {
              result[0] += -0.08923702256777122;
            }
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.30829524993896662) ) ) {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)88.00000000000001421) ) ) {
                    result[0] += 0.05736534632572441;
                  } else {
                    result[0] += -0.002500587378134347;
                  }
                } else {
                  result[0] += -0.05886776314623396;
                }
              } else {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += -0.04824445948734251;
                } else {
                  result[0] += 0.004577689746285032;
                }
              }
            } else {
              result[0] += 0.025752089244983834;
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.827801465988160068) ) ) {
            if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)13.50000000000000178) ) ) {
              result[0] += -0.009873334005832499;
            } else {
              result[0] += 0.07911994997473025;
            }
          } else {
            result[0] += -0.053671057807327886;
          }
        } else {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
            result[0] += -0.09689846665107206;
          } else {
            result[0] += -0.061091863176885;
          }
        }
      } else {
        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.232423543930054599) ) ) {
          if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)13.50000000000000178) ) ) {
            result[0] += 0.0005697938661339824;
          } else {
            result[0] += 0.06438224814705375;
          }
        } else {
          result[0] += -0.0414141441000466;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)16.50000000000000355) ) ) {
      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.595119953155518466) ) ) {
          result[0] += 0.08740095932318903;
        } else {
          result[0] += -0.07590778839693396;
        }
      } else {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
          result[0] += -0.030334636633499515;
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.744781017303467685) ) ) {
            result[0] += -0.006990808863086582;
          } else {
            result[0] += 0.053714975164885284;
          }
        }
      }
    } else {
      result[0] += -0.0781506813673987;
    }
  }
  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
            if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
              if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                result[0] += 0.00917189801746421;
              } else {
                result[0] += -0.08676920284481546;
              }
            } else {
              result[0] += 0.03534534698338668;
            }
          } else {
            result[0] += 0.07146430834388216;
          }
        } else {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.65486812591552912) ) ) {
            if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.09773197940339484;
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.827801465988160068) ) ) {
                result[0] += 0.01776957051994538;
              } else {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)80.00000000000001421) ) ) {
                  result[0] += -0.014197598002938891;
                } else {
                  result[0] += 0.019623226722789353;
                }
              }
            }
          } else {
            result[0] += 0.04940723527999823;
          }
        }
      } else {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.516936540603638583) ) ) {
            result[0] += -0.08160603628811613;
          } else {
            if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.827801465988160068) ) ) {
                result[0] += -0.09106636612289848;
              } else {
                result[0] += 0.005077430858869364;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.481347560882569248) ) ) {
                result[0] += 0.014965118319496724;
              } else {
                result[0] += -0.08876692066761674;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
            if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)7.000000000000000888) ) ) {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.48918962478637873) ) ) {
                  if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)137422176256.0000153) ) ) {
                    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.07453290364446281;
                    } else {
                      if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += 0.006444090678954655;
                      } else {
                        result[0] += 0.058898043911149014;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.04969787597656428) ) ) {
                      if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)3.000000000000000444) ) ) {
                        result[0] += 0.08242542159588179;
                      } else {
                        if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.671854496002199042) ) ) {
                            result[0] += 0.02258833815964364;
                          } else {
                            result[0] += -0.02858508052280631;
                          }
                        } else {
                          result[0] += -0.07587581245805414;
                        }
                      }
                    } else {
                      result[0] += -0.06475617014872889;
                    }
                  }
                } else {
                  result[0] += -0.08271588850557987;
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.827801465988160068) ) ) {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += 0.007095451923455446;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.055836200714113104) ) ) {
                      result[0] += -0.02390748676732395;
                    } else {
                      if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += 0.02694598580861939;
                      } else {
                        result[0] += 0.06615788218713464;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)137422176256.0000153) ) ) {
                    if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                      if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)3072.000000000000455) ) ) {
                        result[0] += 0.03327875598996788;
                      } else {
                        result[0] += -0.08406875581980151;
                      }
                    } else {
                      result[0] += 0.026689935093373703;
                    }
                  } else {
                    if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.07482418035134812;
                    } else {
                      result[0] += -0.012011358490715529;
                    }
                  }
                }
              }
            } else {
              result[0] += -0.07584642550537747;
            }
          } else {
            if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += -0.03798044607770646;
              } else {
                result[0] += 0.01391295289237659;
              }
            } else {
              if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)2.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.97438240051269709) ) ) {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += 0.07522869969792068;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.807813167572023261) ) ) {
                      result[0] += -0.1333614437777427;
                    } else {
                      result[0] += 0.03631368706739187;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                    if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += -0.04097412343238205;
                    } else {
                      result[0] += 0.06383914020140638;
                    }
                  } else {
                    result[0] += 0.039674022295511216;
                  }
                }
              } else {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.834949493408204901) ) ) {
                    result[0] += -0.0810194899999338;
                  } else {
                    result[0] += 0.02343178925564192;
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
                    result[0] += 0.050458913010519336;
                  } else {
                    result[0] += -0.08318982104192228;
                  }
                }
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.154959201812744585) ) ) {
            result[0] += 0.06606856646019076;
          } else {
            result[0] += -0.03026196675889364;
          }
        } else {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
            result[0] += -0.09610155484203009;
          } else {
            result[0] += -0.057738410118996865;
          }
        }
      } else {
        if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)5.500000000000000888) ) ) {
          if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.694163918495178667) ) ) {
              result[0] += -0.019133507564405912;
            } else {
              result[0] += -0.08934926637676324;
            }
          } else {
            result[0] += 0.006838159233033256;
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.516936540603638583) ) ) {
            result[0] += 0.06192861374716287;
          } else {
            result[0] += -0.03678313150564409;
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.595119953155518466) ) ) {
        result[0] += 0.08113108451721537;
      } else {
        result[0] += -0.07637980129673373;
      }
    } else {
      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
          result[0] += -0.0928259599025992;
        } else {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.695749998092652255) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.329314231872559482) ) ) {
              result[0] += -0.08300958809128539;
            } else {
              result[0] += 0.03970643642004855;
            }
          } else {
            result[0] += 0.03754339271834354;
          }
        }
      } else {
        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.605039834976196733) ) ) {
          result[0] += -0.034244101907932055;
        } else {
          if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)17.50000000000000355) ) ) {
            result[0] += 0.005024251526689495;
          } else {
            result[0] += 0.05209250069203816;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
          if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
            if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.022264651634568155;
                } else {
                  result[0] += -0.03186243289929547;
                }
              } else {
                result[0] += 0.051440193959948255;
              }
            } else {
              result[0] += -0.08734514746435006;
            }
          } else {
            result[0] += 0.033971613092391925;
          }
        } else {
          if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
            result[0] += 0.07841732539205418;
          } else {
            result[0] += 0.012549460420564288;
          }
        }
      } else {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.65486812591552912) ) ) {
              result[0] += 0.001833800476796332;
            } else {
              result[0] += 0.04591349120113739;
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.778982400894165927) ) ) {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += -0.08490282271742018;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.669892311096192294) ) ) {
                  result[0] += 0.04831816811767992;
                } else {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                    result[0] += -0.018418220164406647;
                  } else {
                    result[0] += -0.0661721327810046;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.807813167572023261) ) ) {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)88.00000000000001421) ) ) {
                    if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
                      if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                        result[0] += -0.05028642637835308;
                      } else {
                        result[0] += 0.03845088296525732;
                      }
                    } else {
                      result[0] += 0.09795245223619811;
                    }
                  } else {
                    result[0] += -0.05666866268292087;
                  }
                } else {
                  if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += -0.020422125223122308;
                  } else {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += -0.07919645174275794;
                    } else {
                      result[0] += 8.061897612685462e-05;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.07162572068150369;
                } else {
                  if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)5.500000000000000888) ) ) {
                    if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)3.500000000000000444) ) ) {
                      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)80.00000000000001421) ) ) {
                        if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)21.50000000000000355) ) ) {
                          result[0] += -0.063122048450337;
                        } else {
                          result[0] += 0.05087539416869369;
                        }
                      } else {
                        result[0] += 0.03587957848016217;
                      }
                    } else {
                      result[0] += 0.11156813212705011;
                    }
                  } else {
                    if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)30.50000000000000355) ) ) {
                      result[0] += -0.09037123196527133;
                    } else {
                      result[0] += 0.01116326370648449;
                    }
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
            if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -7.98221575004262e-05;
            } else {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)88.00000000000001421) ) ) {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += 0.00041419777062836157;
                  } else {
                    result[0] += 0.03594107101743491;
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.778982400894165927) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.978769779205324042) ) ) {
                      result[0] += -0.02917807644278267;
                    } else {
                      result[0] += 0.07310437993767514;
                    }
                  } else {
                    result[0] += -0.010531682022738437;
                  }
                }
              } else {
                result[0] += -0.020251359592788884;
              }
            }
          } else {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
              result[0] += -0.08290804367310088;
            } else {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)11.50000000000000178) ) ) {
                  if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)6.500000000000000888) ) ) {
                    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)56.00000000000000711) ) ) {
                      result[0] += -0.04773616286797539;
                    } else {
                      if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                            result[0] += -0.17298894967285722;
                          } else {
                            result[0] += 0.041401351307103304;
                          }
                        } else {
                          result[0] += -0.03520304314475862;
                        }
                      } else {
                        if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)7.000000000000000888) ) ) {
                          result[0] += -0.04140707103741198;
                        } else {
                          result[0] += 0.04083181671499589;
                        }
                      }
                    }
                  } else {
                    result[0] += 0.033330685069105824;
                  }
                } else {
                  result[0] += -0.06414921583153449;
                }
              } else {
                result[0] += 0.13543251820911212;
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)5.500000000000000888) ) ) {
        if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.694163918495178667) ) ) {
            result[0] += -0.05791386410512126;
          } else {
            result[0] += -0.09448507766398624;
          }
        } else {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                result[0] += 0.06320411219770064;
              } else {
                result[0] += -0.03423185142086521;
              }
            } else {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += -0.10581119768409658;
              } else {
                result[0] += -0.03965431409221185;
              }
            }
          } else {
            result[0] += 0.003473116921034973;
          }
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.827801465988160068) ) ) {
          if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)13.50000000000000178) ) ) {
            result[0] += -0.01141327725340323;
          } else {
            result[0] += 0.07155348902003748;
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.827801465988160068) ) ) {
            if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)7.500000000000000888) ) ) {
              result[0] += 0.10410120848981322;
            } else {
              result[0] += -0.042436622550711914;
            }
          } else {
            result[0] += -0.0658474508388831;
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.970608234405518466) ) ) {
        result[0] += 0.08133276789589819;
      } else {
        result[0] += -0.08122326698296962;
      }
    } else {
      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
        result[0] += -0.09102357280333855;
      } else {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.718933820724488193) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.849175214767456943) ) ) {
              result[0] += -0.10860522511283177;
            } else {
              result[0] += 0.018228560810393205;
            }
          } else {
            result[0] += 0.02853161787674341;
          }
        } else {
          result[0] += 0.04160535260234134;
        }
      }
    }
  }
  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
          if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.084203958511353427) ) ) {
              result[0] += -0.13268546820529936;
            } else {
              result[0] += 0.08554277872643992;
            }
          } else {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
              result[0] += 0.06423361049437547;
            } else {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
                result[0] += -0.004608348339309854;
              } else {
                result[0] += 0.04817190365890199;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.994492053985595925) ) ) {
            result[0] += 0.01992624228070161;
          } else {
            result[0] += 0.06145164943163012;
          }
        }
      } else {
        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
          if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.500000000000000444) ) ) {
              if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.061727773772745576;
                } else {
                  result[0] += -0.02248910760429458;
                }
              } else {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
                  result[0] += -0.006709360390475785;
                } else {
                  result[0] += -0.10051652513582043;
                }
              }
            } else {
              if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)8.500000000000001776) ) ) {
                if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)6.500000000000000888) ) ) {
                  result[0] += -0.016689020941959187;
                } else {
                  if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                      result[0] += 0.06775842369325118;
                    } else {
                      result[0] += -0.10695183268566964;
                    }
                  } else {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += 0.07017312365826921;
                    } else {
                      result[0] += -0.03968904259819259;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)21.50000000000000355) ) ) {
                  result[0] += -0.026906745088755287;
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.90474271774292081) ) ) {
                    result[0] += 0.023047599131814797;
                  } else {
                    result[0] += -0.1071786269132305;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
              if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += -0.05598937870123219;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.1822080612182635) ) ) {
                  if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)3.500000000000000444) ) ) {
                    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                      result[0] += -0.004266013646559446;
                    } else {
                      if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
                        result[0] += 0.01023975622631045;
                      } else {
                        result[0] += 0.06170253312008278;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
                      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                        result[0] += 0.023648229705769207;
                      } else {
                        result[0] += -0.07542302434224965;
                      }
                    } else {
                      result[0] += -0.06599511026115185;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                    if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)7.500000000000000888) ) ) {
                      result[0] += 0.05494978699407174;
                    } else {
                      result[0] += -0.11155617664600888;
                    }
                  } else {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += -0.039816127082943964;
                    } else {
                      result[0] += 0.023154094059626276;
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.035399569775471165;
              } else {
                if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += -0.057734264571432825;
                } else {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)80.00000000000001421) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.484580039978028232) ) ) {
                      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.65882921218872248) ) ) {
                            result[0] += 0.007347836228562656;
                          } else {
                            result[0] += 0.05936774432707442;
                          }
                        } else {
                          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                            result[0] += -0.04243478493565228;
                          } else {
                            result[0] += 0.02010005731041594;
                          }
                        }
                      } else {
                        result[0] += 0.0705742557722169;
                      }
                    } else {
                      if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)4.500000000000000888) ) ) {
                        result[0] += -0.03097841632690672;
                      } else {
                        if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                          result[0] += -0.051195070596319205;
                        } else {
                          if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                            if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.039720773696899636) ) ) {
                              result[0] += -0.04678748182575738;
                            } else {
                              result[0] += 0.04847501386475356;
                            }
                          } else {
                            if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                              result[0] += -0.0990354499085772;
                            } else {
                              result[0] += 0.054625637548155775;
                            }
                          }
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.484580039978028232) ) ) {
                      result[0] += -0.045080962604170605;
                    } else {
                      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.172047138214112216) ) ) {
                        if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.44140100479126021) ) ) {
                          if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)7.000000000000000888) ) ) {
                            result[0] += 0.05029723018716176;
                          } else {
                            result[0] += -0.01677287419046961;
                          }
                        } else {
                          result[0] += -0.09235166756997608;
                        }
                      } else {
                        result[0] += 0.0892291367788956;
                      }
                    }
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.39281225204467951) ) ) {
              result[0] += 0.028328110417389165;
            } else {
              result[0] += -0.060503466197770854;
            }
          } else {
            result[0] += -0.06961233703585194;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)5.500000000000000888) ) ) {
        if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
          result[0] += -0.07665926572888278;
        } else {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.0012576680702090235;
            } else {
              result[0] += -0.06391136466463847;
            }
          } else {
            result[0] += 0.0007045026181038762;
          }
        }
      } else {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.467917680740357333) ) ) {
          if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)11.50000000000000178) ) ) {
            result[0] += 0.0556847030871939;
          } else {
            result[0] += -0.05414376212083869;
          }
        } else {
          result[0] += -0.05524605443622003;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
      result[0] += 0.0760838893473534;
    } else {
      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
        result[0] += -0.08937044417739504;
      } else {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.605039834976196733) ) ) {
          result[0] += -0.06658728870993426;
        } else {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
            result[0] += 0.007866200412118505;
          } else {
            result[0] += 0.043933956718255895;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
          result[0] += 0.015550504585563089;
        } else {
          if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.497866153717041238) ) ) {
              result[0] += 0.03840779869332389;
            } else {
              result[0] += 0.0999912546523495;
            }
          } else {
            result[0] += 0.007438180810972673;
          }
        }
      } else {
        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
          if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.500000000000000444) ) ) {
              if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.05951791840913043;
                } else {
                  result[0] += -0.021986248866368643;
                }
              } else {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)120.0000000000000142) ) ) {
                  result[0] += -0.0055707843530354096;
                } else {
                  result[0] += -0.0989266211418176;
                }
              }
            } else {
              if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2202.000000000000455) ) ) {
                result[0] += -0.11346558655322499;
              } else {
                if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)8.500000000000001776) ) ) {
                  if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)6.500000000000000888) ) ) {
                    result[0] += -0.014167460138132216;
                  } else {
                    result[0] += 0.039540262725637064;
                  }
                } else {
                  if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)9.500000000000001776) ) ) {
                    result[0] += -0.08084985548318192;
                  } else {
                    if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)14.50000000000000178) ) ) {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.90474271774292081) ) ) {
                        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += 0.04704336317339245;
                        } else {
                          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                            result[0] += -0.08576218902989315;
                          } else {
                            result[0] += 0.0019534901099485897;
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)19.50000000000000355) ) ) {
                          result[0] += 0.056626100922778005;
                        } else {
                          result[0] += -0.06261788692832161;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.255632162094117099) ) ) {
                        result[0] += -0.027812906312561492;
                      } else {
                        result[0] += 0.08863230742290272;
                      }
                    }
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
              if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += -0.05495516330164344;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.1822080612182635) ) ) {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += -0.012027591219659876;
                  } else {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                      if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)7.000000000000000888) ) ) {
                        result[0] += 0.03934517083443912;
                      } else {
                        result[0] += -0.07975813351801295;
                      }
                    } else {
                      result[0] += -0.026781836008441823;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                    if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)7.500000000000000888) ) ) {
                      if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += -0.12548281406451403;
                      } else {
                        result[0] += 0.05822636762271062;
                      }
                    } else {
                      result[0] += -0.09995429427912966;
                    }
                  } else {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += -0.016825733932034657;
                        } else {
                          result[0] += -0.06851703170734416;
                        }
                      } else {
                        result[0] += -0.011343890603476034;
                      }
                    } else {
                      result[0] += 0.021302488812199802;
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                  result[0] += 0.09264769287626723;
                } else {
                  result[0] += -0.041357775977467384;
                }
              } else {
                if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += -0.05352044471394064;
                } else {
                  if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                    result[0] += 0.01628611325365685;
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.484580039978028232) ) ) {
                      result[0] += -0.08922642055986577;
                    } else {
                      result[0] += 0.09756398547483754;
                    }
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.39281225204467951) ) ) {
              result[0] += 0.025125167879879146;
            } else {
              result[0] += -0.05971940288997583;
            }
          } else {
            result[0] += -0.06705057479055944;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)5.500000000000000888) ) ) {
        if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.694163918495178667) ) ) {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.021239949817874595;
              } else {
                result[0] += -0.09464966062279878;
              }
            } else {
              result[0] += -0.011136828705525997;
            }
          } else {
            result[0] += -0.09250646261184856;
          }
        } else {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                result[0] += 0.05598333477979091;
              } else {
                result[0] += -0.03389912610058818;
              }
            } else {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += -0.10376058836830984;
              } else {
                result[0] += -0.03694646662059891;
              }
            }
          } else {
            result[0] += -0.0010139868154797596;
          }
        }
      } else {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.607751369476319248) ) ) {
          if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)11.50000000000000178) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.993164777755738193) ) ) {
              result[0] += 0.05828215222274559;
            } else {
              result[0] += -0.03528096915895733;
            }
          } else {
            result[0] += -0.053628075953155774;
          }
        } else {
          result[0] += -0.058944347886512885;
        }
      }
    }
  } else {
    if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.970608234405518466) ) ) {
        if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)16.50000000000000355) ) ) {
          result[0] += 0.08008912854109708;
        } else {
          result[0] += -0.08107498495641241;
        }
      } else {
        result[0] += -0.11393840810595848;
      }
    } else {
      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.743881702423096591) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.534971714019776279) ) ) {
            result[0] += -0.08097393719054581;
          } else {
            result[0] += 0.025574943861304263;
          }
        } else {
          result[0] += 0.014346762309816116;
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.744781017303467685) ) ) {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.607751369476319248) ) ) {
              result[0] += -0.11270506757583668;
            } else {
              result[0] += 0.03077668541201005;
            }
          } else {
            result[0] += 0.039940316155537975;
          }
        } else {
          result[0] += 0.04130541288966829;
        }
      }
    }
  }
  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.79870843887329279) ) ) {
        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)20.00000000000000355) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.671854496002199042) ) ) {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)120.0000000000000142) ) ) {
              result[0] += 0.003094284537414875;
            } else {
              result[0] += -0.11398696274994938;
            }
          } else {
            if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)9.500000000000001776) ) ) {
              if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)120.0000000000000142) ) ) {
                  result[0] += 0.035781356748644334;
                } else {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.06204946963216926;
                  } else {
                    result[0] += 0.019982285581117892;
                  }
                }
              } else {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.484580039978028232) ) ) {
                    result[0] += 0.014641026723671069;
                  } else {
                    if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)56.00000000000000711) ) ) {
                      result[0] += -0.08788050322553725;
                    } else {
                      result[0] += -0.008508809404187611;
                    }
                  }
                } else {
                  result[0] += 0.07361245759329078;
                }
              }
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.607751369476319248) ) ) {
                result[0] += 0.016687945614496293;
              } else {
                result[0] += -0.06456529955209402;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.888826131820679155) ) ) {
            result[0] += 0.013204101233601363;
          } else {
            if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.06712246424363487;
            } else {
              if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
                if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += -0.08775832340431282;
                } else {
                  if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)2.500000000000000444) ) ) {
                    result[0] += 0.04135027251205024;
                  } else {
                    result[0] += -0.021532559564665304;
                  }
                }
              } else {
                if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)7.000000000000000888) ) ) {
                  if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                    result[0] += 0.09836466164215477;
                  } else {
                    result[0] += -0.019568520335681114;
                  }
                } else {
                  result[0] += -0.08110058048945716;
                }
              }
            }
          }
        }
      } else {
        result[0] += 0.03969644093201124;
      }
    } else {
      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.172047138214112216) ) ) {
          if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
            result[0] += -0.061324421641875106;
          } else {
            result[0] += -0.09619704032585279;
          }
        } else {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.012675821781158891) ) ) {
            if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += 0.054982594114252964;
            } else {
              result[0] += -0.03112855232737198;
            }
          } else {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.516936540603638583) ) ) {
                result[0] += -0.061708420783942175;
              } else {
                result[0] += -0.00033440310769035;
              }
            } else {
              result[0] += -0.08849349735142434;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.04969787597656428) ) ) {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                    result[0] += 0.11078519234362283;
                  } else {
                    result[0] += -0.04698255267946345;
                  }
                } else {
                  result[0] += -0.06105710880479927;
                }
              } else {
                if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.10492117641783333;
                } else {
                  if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)11.50000000000000178) ) ) {
                    if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
                      result[0] += 0.019084231581664165;
                    } else {
                      result[0] += -0.07784150461434139;
                    }
                  } else {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.825422286987305576) ) ) {
                      if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)21.50000000000000355) ) ) {
                        result[0] += -0.055196534985023575;
                      } else {
                        result[0] += 0.050215976636508244;
                      }
                    } else {
                      if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)31.50000000000000355) ) ) {
                        result[0] += -0.10401504903611579;
                      } else {
                        result[0] += 0.03296387625919846;
                      }
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.19876670837402521) ) ) {
                if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.025024000592401938;
                  } else {
                    result[0] += 0.036785751854998265;
                  }
                } else {
                  result[0] += -0.048975911492280894;
                }
              } else {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.07187192416709214;
                } else {
                  result[0] += -0.0296576592925785;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.827801465988160068) ) ) {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.79870843887329279) ) ) {
                  result[0] += 0.004389349017727542;
                } else {
                  result[0] += -0.047631083307873356;
                }
              } else {
                if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.59331560134887873) ) ) {
                    result[0] += -0.1569677887655962;
                  } else {
                    result[0] += 0.03899117474573257;
                  }
                } else {
                  if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)13.50000000000000178) ) ) {
                    result[0] += 0.02557715576087054;
                  } else {
                    result[0] += 0.06155795300975347;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                  result[0] += -0.058802535243816345;
                } else {
                  if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.500000000000000444) ) ) {
                      result[0] += -0.03053902223263415;
                    } else {
                      result[0] += 0.02868948493391901;
                    }
                  } else {
                    result[0] += 0.05703359105076458;
                  }
                }
              } else {
                if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
                  result[0] += -0.04715039047114473;
                } else {
                  result[0] += 0.01045476490043668;
                }
              }
            }
          }
        } else {
          result[0] += 0.016815526647587436;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)16.50000000000000355) ) ) {
        result[0] += 0.07501776143286057;
      } else {
        result[0] += -0.07837847094805674;
      }
    } else {
      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.827801465988160068) ) ) {
          result[0] += -0.13512024801691097;
        } else {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
            result[0] += -0.0823956463523473;
          } else {
            result[0] += 0.0027722595034145882;
          }
        }
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.284418344497681552) ) ) {
          if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
            result[0] += 0.023276024514538377;
          } else {
            result[0] += -0.0758718704030889;
          }
        } else {
          result[0] += 0.039497022981785375;
        }
      }
    }
  }
}

