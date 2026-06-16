
#include "header.h"

void predict_unit2(union Entry* data, double* result) {
  unsigned int tmp;
  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
    if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
        if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += 0.003634678994758537;
        } else {
          if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.484580039978028232) ) ) {
                result[0] += 0.004542403799570623;
              } else {
                if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.124530076980591708) ) ) {
                    result[0] += -0.027335618337572972;
                  } else {
                    result[0] += -0.10135397674911986;
                  }
                } else {
                  if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                    if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.531007289886475498) ) ) {
                        result[0] += -0.020603506774902053;
                      } else {
                        result[0] += -0.09796603570711895;
                      }
                    } else {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.801661729812622958) ) ) {
                        result[0] += -0.0018283019960561892;
                      } else {
                        result[0] += 0.06182371901915362;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                      if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                        result[0] += -0.029881023141584404;
                      } else {
                        result[0] += 0.015958142427089754;
                      }
                    } else {
                      if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                        result[0] += 0.03332964063825602;
                      } else {
                        result[0] += -0.007698436518050307;
                      }
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                  if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.027214508770099288;
                    } else {
                      result[0] += -0.08424778902999906;
                    }
                  } else {
                    result[0] += -0.01367620959014105;
                  }
                } else {
                  if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.01664493885905084;
                  } else {
                    result[0] += 0.03207306383523631;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.142630577087403232) ) ) {
                  result[0] += 0.05361932863845817;
                } else {
                  if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.070054531097412998) ) ) {
                      if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
                        result[0] += -0.025518547690198887;
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
                          if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                            result[0] += -0.09732169062113254;
                          } else {
                            result[0] += 0.01976197899378284;
                          }
                        } else {
                          if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                            result[0] += 0.04594691884914881;
                          } else {
                            result[0] += 0.0008761232956111084;
                          }
                        }
                      }
                    } else {
                      result[0] += -0.0786265965604606;
                    }
                  } else {
                    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
                        result[0] += -0.04327376077896474;
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.397998809814454013) ) ) {
                          result[0] += 0.11955185935332353;
                        } else {
                          if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                            result[0] += 0.07887339800809232;
                          } else {
                            result[0] += 0.015750759357972304;
                          }
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.172047138214112216) ) ) {
                        result[0] += -0.027699064710426748;
                      } else {
                        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
                          if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                            result[0] += 0.0351278428704107;
                          } else {
                            result[0] += -0.012176314591126669;
                          }
                        } else {
                          result[0] += 0.07938150131402473;
                        }
                      }
                    }
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.10472718369407144;
              } else {
                result[0] += -0.016790303565824016;
              }
            } else {
              if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.605039834976196733) ) ) {
                  result[0] += 0.015766208109901;
                } else {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += -0.017442477923431093;
                  } else {
                    result[0] += -0.06436502063705654;
                  }
                }
              } else {
                result[0] += 0.029671019867061728;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
          if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
            result[0] += -0.023534799735347504;
          } else {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
              result[0] += 0.10678170990822496;
            } else {
              result[0] += -0.07760432161040176;
            }
          }
        } else {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.433569431304932529) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.029068946838379794) ) ) {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.053125854316907764;
              } else {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                  if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5.336091518402100498) ) ) {
                    result[0] += 0.10471101447933875;
                  } else {
                    result[0] += -0.07694804839574358;
                  }
                } else {
                  result[0] += -0.03615341861847525;
                }
              }
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.602003335952759233) ) ) {
                result[0] += 0.05163396005030078;
              } else {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += -0.08522692979414243;
                } else {
                  result[0] += -0.0094089115266352;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
              result[0] += 0.00048220948387887554;
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.617236852645874912) ) ) {
                if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.938867926597595659) ) ) {
                  result[0] += -0.12391709849275441;
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.431880712509156162) ) ) {
                    result[0] += 0.06262178478017284;
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.172047138214112216) ) ) {
                      result[0] += -0.12253637444216624;
                    } else {
                      result[0] += 0.08671037827467296;
                    }
                  }
                }
              } else {
                result[0] += 0.1352913154269077;
              }
            }
          }
        }
      }
    } else {
      result[0] += -0.02845162409706671;
    }
  } else {
    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
        result[0] += -0.0026477624767288427;
      } else {
        if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
            result[0] += 0.1572998143320989;
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
              result[0] += -0.05929475327874284;
            } else {
              result[0] += 0.1505107622801652;
            }
          }
        } else {
          result[0] += -0.09050281291199268;
        }
      }
    } else {
      if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)6.000000000000000888) ) ) {
        result[0] += -0.0793756003594027;
      } else {
        if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
          result[0] += -0.049554710082098254;
        } else {
          result[0] += -0.011566137225225884;
        }
      }
    }
  }
  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
      if ( UNLIKELY(  (data[43].missing != -1) && (data[43].fvalue <= (double)-1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
          if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.835998296737671787) ) ) {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)46.00000000000000711) ) ) {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += 0.00040963915313196094;
                } else {
                  result[0] += -0.03343811865580148;
                }
              } else {
                result[0] += 0.011388407704896793;
              }
            } else {
              result[0] += -0.005661606140313861;
            }
          } else {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += 0.0415246668461851;
            } else {
              result[0] += -0.07338206302728735;
            }
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
              result[0] += 0.012170135209546747;
            } else {
              result[0] += -0.09700239604919608;
            }
          } else {
            if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
                  result[0] += 0.045713354531194246;
                } else {
                  result[0] += -0.09552579054307402;
                }
              } else {
                result[0] += -0.06708542229541267;
              }
            } else {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                result[0] += 0.0027456440988100467;
              } else {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.651049375534058505) ) ) {
                    result[0] += 0.0746835057372591;
                  } else {
                    result[0] += -0.01664185241053821;
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.599987030029298651) ) ) {
                    result[0] += -0.08480172962701643;
                  } else {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.142630577087403232) ) ) {
                      result[0] += -0.011881219496817728;
                    } else {
                      result[0] += 0.057628776971459864;
                    }
                  }
                }
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.070054531097412998) ) ) {
              if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.802696108818054643) ) ) {
                    result[0] += 0.0022373679703823666;
                  } else {
                    result[0] += -0.04781934668009883;
                  }
                } else {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += -0.013129263191925509;
                  } else {
                    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                      result[0] += 0.0014693886699307527;
                    } else {
                      if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += -0.029860089527471004;
                      } else {
                        result[0] += 0.04397847397011192;
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                  result[0] += -0.018289516276493674;
                } else {
                  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.249904870986938921) ) ) {
                      result[0] += 0.06294349274801485;
                    } else {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.524927973747253862) ) ) {
                        result[0] += -0.09165616327652762;
                      } else {
                        result[0] += -0.007661640786762684;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.651049375534058505) ) ) {
                      result[0] += 0.04869997431115151;
                    } else {
                      result[0] += 0.012314213244339928;
                    }
                  }
                }
              }
            } else {
              result[0] += -0.03402408231167871;
            }
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.651049375534058505) ) ) {
              if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                result[0] += -0.059714141150917326;
              } else {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                  result[0] += -0.03058827840601467;
                } else {
                  result[0] += 0.007826788674167579;
                }
              }
            } else {
              if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += -0.07806902292441957;
              } else {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
                  result[0] += -0.0175774990412285;
                } else {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += -0.026026648137837134;
                  } else {
                    result[0] += 0.019643927582618015;
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += -0.002503220873933288;
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.141444921493531162) ) ) {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)40.00000000000000711) ) ) {
                      result[0] += 0.08814245244077185;
                    } else {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.484580039978028232) ) ) {
                        result[0] += -0.02550489432451334;
                      } else {
                        result[0] += 0.024980644757684398;
                      }
                    }
                  } else {
                    result[0] += -0.04285466361763516;
                  }
                }
              } else {
                result[0] += 0.03595945141760815;
              }
            } else {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                result[0] += -0.08428107354790393;
              } else {
                result[0] += -0.0007036451007147549;
              }
            }
          } else {
            if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.172047138214112216) ) ) {
                result[0] += 0.03807755819099369;
              } else {
                result[0] += -0.04443507619150378;
              }
            } else {
              result[0] += -0.039506739918987094;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.842307567596437323) ) ) {
        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
          result[0] += 0.07188047951290699;
        } else {
          result[0] += -0.0602791250340886;
        }
      } else {
        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.88435244560241788) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)176.0000000000000284) ) ) {
                result[0] += 0.05426818009576764;
              } else {
                result[0] += -0.09716852034450243;
              }
            } else {
              result[0] += -0.059567194566536565;
            }
          } else {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += -0.07635402158717285;
            } else {
              result[0] += -0.00017105469095244057;
            }
          }
        } else {
          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
            if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += 0.02183361154198676;
            } else {
              result[0] += -0.10288221656154523;
            }
          } else {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += -0.03358632842922742;
              } else {
                result[0] += 0.11469967169003886;
              }
            } else {
              result[0] += 0.08080908814720983;
            }
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
      result[0] += -0.0029058436811374966;
    } else {
      if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)6.000000000000000888) ) ) {
        result[0] += -0.07739028525231925;
      } else {
        result[0] += -0.018559234581665352;
      }
    }
  }
  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
        if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += 0.003022057331791741;
        } else {
          if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
              if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.484580039978028232) ) ) {
                  result[0] += 0.0034049221957643803;
                } else {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += -0.030679912672617972;
                  } else {
                    result[0] += 0.0015131297708573168;
                  }
                }
              } else {
                result[0] += 0.008647707409091007;
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)40.00000000000000711) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
                    result[0] += 0.06000875445271862;
                  } else {
                    result[0] += -0.038464320189116566;
                  }
                } else {
                  if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                    if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += -0.07162939562181458;
                    } else {
                      result[0] += -0.015579532819470585;
                    }
                  } else {
                    result[0] += -0.00635083900122537;
                  }
                }
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.255632162094117099) ) ) {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.142630577087403232) ) ) {
                    result[0] += 0.043380337188205886;
                  } else {
                    if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                      result[0] += 0.001569741497979546;
                    } else {
                      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                        result[0] += 0.045214085114165414;
                      } else {
                        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.90638875961303889) ) ) {
                            result[0] += -0.02224073529651102;
                          } else {
                            result[0] += 0.039129557168278664;
                          }
                        } else {
                          result[0] += -0.03565675900751485;
                        }
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                    result[0] += -0.05903842950048941;
                  } else {
                    result[0] += 0.042519046698438964;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.1029047581428173;
              } else {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.484580039978028232) ) ) {
                    result[0] += 0.02159577255060626;
                  } else {
                    result[0] += -0.06479348214558316;
                  }
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.232423543930054599) ) ) {
                    result[0] += -0.07673635454076154;
                  } else {
                    result[0] += 0.01750865112736843;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.605039834976196733) ) ) {
                if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.026314011722239905;
                } else {
                  result[0] += -0.0771565104411589;
                }
              } else {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += -0.015628211940371953;
                  } else {
                    result[0] += -0.06103895060122133;
                  }
                } else {
                  result[0] += 0.02396562509418623;
                }
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
          result[0] += -0.040903549027556496;
        } else {
          result[0] += 0.024982092497494304;
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.842307567596437323) ) ) {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
          result[0] += 0.09078219540231522;
        } else {
          result[0] += 0.010845989511889219;
        }
      } else {
        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.88435244560241788) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
            if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.651049375534058505) ) ) {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += 0.07426996506407656;
                } else {
                  result[0] += -0.06270080775556412;
                }
              } else {
                if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.007969518052735259;
                } else {
                  result[0] += 0.10060509295912307;
                }
              }
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.45958471298217951) ) ) {
                result[0] += 0.07700772063070598;
              } else {
                result[0] += -0.10451458067507897;
              }
            }
          } else {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
              if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                result[0] += 0.014199960957213254;
              } else {
                result[0] += -0.10227783649405092;
              }
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.284418344497681552) ) ) {
                result[0] += -0.10763538516578393;
              } else {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                  result[0] += -0.01715642753829225;
                } else {
                  result[0] += 0.03700122512189654;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
            if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += 0.020369500863305745;
            } else {
              result[0] += -0.09840608143800508;
            }
          } else {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += -0.033169973412293816;
              } else {
                result[0] += 0.10555725610742009;
              }
            } else {
              result[0] += 0.07238447321352072;
            }
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
        if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
          result[0] += -0.05242075924736235;
        } else {
          result[0] += 0.18595478390819653;
        }
      } else {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
          result[0] += -0.0023129506822504773;
        } else {
          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += 0.11837132199622823;
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += -0.057242964005926635;
              } else {
                result[0] += 0.11552961851520911;
              }
            }
          } else {
            result[0] += -0.0894259567640852;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)6.000000000000000888) ) ) {
        result[0] += -0.0759293832261704;
      } else {
        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)9.500000000000001776) ) ) {
          if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += -0.08428992598157123;
          } else {
            if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
              result[0] += -0.0020674414184153472;
            } else {
              result[0] += -0.03462891760462007;
            }
          }
        } else {
          if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)10.50000000000000178) ) ) {
            if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += -0.027472651788754424;
              } else {
                result[0] += 0.08214417789655949;
              }
            } else {
              result[0] += 0.16718622412703277;
            }
          } else {
            result[0] += -0.04521173098133477;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
    if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY(  (data[43].missing != -1) && (data[43].fvalue <= (double)-1.00000001800250948e-35) ) ) {
        result[0] += 0.0035274193907317386;
      } else {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.142630577087403232) ) ) {
          result[0] += -0.011547831261087972;
        } else {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
              if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += -0.02038629173700887;
              } else {
                result[0] += -0.001158667820826636;
              }
            } else {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += 0.0043037876892964375;
              } else {
                result[0] += 0.07093720813687766;
              }
            }
          } else {
            result[0] += 0.01630919416305068;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
        if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.699081301689148393) ) ) {
            result[0] += 0.03355442729647167;
          } else {
            result[0] += -0.031108457670597;
          }
        } else {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)14.74696540832519709) ) ) {
            result[0] += 0.005267308568457543;
          } else {
            if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += 0.12233682110684344;
            } else {
              result[0] += -0.00047487601480457445;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.142747402191162998) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.397998809814454013) ) ) {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
              if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.835998296737671787) ) ) {
                if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.029068946838379794) ) ) {
                    result[0] += 0.11263095522417968;
                  } else {
                    result[0] += -0.050430104729924576;
                  }
                } else {
                  result[0] += 0.02830353904832473;
                }
              } else {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += 0.024108211897064052;
                } else {
                  result[0] += -0.08280636950007973;
                }
              }
            } else {
              result[0] += -0.04218689556033219;
            }
          } else {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.255632162094117099) ) ) {
              result[0] += -0.04614064514513579;
            } else {
              result[0] += 0.013690074166585976;
            }
          }
        } else {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
            result[0] += -0.017364137512949056;
          } else {
            result[0] += 0.04793268198678428;
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
      if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.978102684020996982) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
            if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2727.500000000000455) ) ) {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += -0.0028487521535433616;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.357691764831543413) ) ) {
                  result[0] += -0.12254179628310685;
                } else {
                  result[0] += 0.02620841458096971;
                }
              }
            } else {
              result[0] += 0.028002284389278532;
            }
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.484580039978028232) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)8816427008.000001907) ) ) {
                  if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)12.00000000000000178) ) ) {
                    result[0] += 0.08820589853215718;
                  } else {
                    result[0] += -0.01659072085912116;
                  }
                } else {
                  result[0] += -9.564459064407406e-05;
                }
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.516936540603638583) ) ) {
                  result[0] += -0.09441081076135183;
                } else {
                  result[0] += 0.11060398594163123;
                }
              }
            } else {
              if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += -0.07852192942451411;
              } else {
                if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += 0.022296587831750078;
                } else {
                  result[0] += -0.02845105648172948;
                }
              }
            }
          }
        } else {
          result[0] += -0.04492700677966933;
        }
      } else {
        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.740319490432739702) ) ) {
          result[0] += 0.03170057140437909;
        } else {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.651049375534058505) ) ) {
              result[0] += 0.01961492359484942;
            } else {
              result[0] += -0.05175580651143277;
            }
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.802696108818054643) ) ) {
              result[0] += -0.04647953444089162;
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.484580039978028232) ) ) {
                    if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2565.000000000000455) ) ) {
                      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                        result[0] += -0.2569251112579079;
                      } else {
                        result[0] += -0.046125109844062234;
                      }
                    } else {
                      result[0] += 0.0775510871209679;
                    }
                  } else {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.597218394279480425) ) ) {
                        result[0] += 0.09867285108072449;
                      } else {
                        result[0] += 0.02888489555546103;
                      }
                    } else {
                      result[0] += -0.11478014507149732;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.029068946838379794) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.607751369476319248) ) ) {
                      result[0] += -0.025782441941245265;
                    } else {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.718933820724488193) ) ) {
                        result[0] += -0.17381624630763076;
                      } else {
                        result[0] += -0.08289268490264595;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.008374855391979345;
                    } else {
                      result[0] += 0.0721232218607887;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.05772382743814397;
                } else {
                  result[0] += 0.01675433044481618;
                }
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)9.500000000000001776) ) ) {
          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
            if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.870983839035034624) ) ) {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
                if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2252.000000000000455) ) ) {
                  result[0] += -0.05044196307337613;
                } else {
                  result[0] += -0.011037402604597056;
                }
              } else {
                result[0] += 0.06986803062832421;
              }
            } else {
              result[0] += -0.04480338957360382;
            }
          } else {
            result[0] += -0.06459594296526892;
          }
        } else {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.861792564392090288) ) ) {
            result[0] += -0.04154857654635619;
          } else {
            result[0] += 0.048451401575173275;
          }
        }
      } else {
        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
          result[0] += 0.07410219310448976;
        } else {
          if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.77496147155761896) ) ) {
              result[0] += 0.06272940929328362;
            } else {
              result[0] += 0.006287304905183433;
            }
          } else {
            result[0] += -0.008989942770773265;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
    if ( UNLIKELY(  (data[42].missing != -1) && (data[42].fvalue <= (double)-1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.597323656082154208) ) ) {
        result[0] += 0.004252277959165795;
      } else {
        result[0] += 0.039751676533296906;
      }
    } else {
      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.302512168884278232) ) ) {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.329314231872559482) ) ) {
            if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.01747802860959786;
            } else {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.05721935548583508;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.7545509338378924) ) ) {
                  if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                    result[0] += -0.03642007465134121;
                  } else {
                    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += -0.018488656796050724;
                    } else {
                      result[0] += 0.032459236193956735;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += 0.011593612824342668;
                  } else {
                    if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                      result[0] += 0.005821310281336479;
                    } else {
                      result[0] += -0.045441167350727715;
                    }
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
              result[0] += -0.04409569565597095;
            } else {
              result[0] += -0.0008213916725884511;
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.92964839935302912) ) ) {
            if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
              result[0] += -0.01759589931149541;
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.970085620880127397) ) ) {
                result[0] += -0.019411294636745857;
              } else {
                result[0] += 0.09156610553478917;
              }
            }
          } else {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.433569431304932529) ) ) {
              result[0] += -0.02753141215759123;
            } else {
              result[0] += -0.06067388054387612;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.284418344497681552) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.215408444404602495) ) ) {
            if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.012362879987169857;
            } else {
              result[0] += 0.04220944005245013;
            }
          } else {
            if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
              if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += -0.1006274945845675;
              } else {
                result[0] += -0.04062092814149226;
              }
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                result[0] += -0.03066459789648071;
              } else {
                result[0] += 0.024455458042644707;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.669892311096192294) ) ) {
              if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.05946990008946218;
              } else {
                result[0] += -0.07610445219208319;
              }
            } else {
              if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.651049375534058505) ) ) {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                      result[0] += 0.10329325127471518;
                    } else {
                      result[0] += -0.03458993982593266;
                    }
                  } else {
                    if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                      result[0] += 0.07606254661315183;
                    } else {
                      result[0] += 0.018193817491926494;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += -0.0726635396209222;
                  } else {
                    if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.497866153717041238) ) ) {
                      result[0] += 0.019637370831277336;
                    } else {
                      result[0] += -0.07068866457881154;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.516936540603638583) ) ) {
                      result[0] += 0.09794354947872999;
                    } else {
                      result[0] += -0.051647028675855015;
                    }
                  } else {
                    result[0] += -0.08820017113000544;
                  }
                } else {
                  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.11962190254490146;
                    } else {
                      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
                          result[0] += -0.06363329915831652;
                        } else {
                          result[0] += 0.06467616209175192;
                        }
                      } else {
                        result[0] += 0.03788546519971617;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.484580039978028232) ) ) {
                      result[0] += -0.033675002272162684;
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.607751369476319248) ) ) {
                        result[0] += 0.1279994326505792;
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.070054531097412998) ) ) {
                          result[0] += -0.0346707359588502;
                        } else {
                          result[0] += 0.09609836067575032;
                        }
                      }
                    }
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.827801465988160068) ) ) {
                  result[0] += 0.09732280193799227;
                } else {
                  result[0] += -0.030724689823223572;
                }
              } else {
                result[0] += 0.020430655340258053;
              }
            } else {
              result[0] += -0.0028795433265141387;
            }
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
      if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
        result[0] += -0.004340982863698051;
      } else {
        if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
          if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
            result[0] += -0.029647968921137902;
          } else {
            result[0] += 0.12820482208672876;
          }
        } else {
          if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.827801465988160068) ) ) {
              result[0] += 0.02646330506156365;
            } else {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                result[0] += -0.058679200358503394;
              } else {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)46.00000000000000711) ) ) {
                  result[0] += 0.04059135549729226;
                } else {
                  result[0] += 0.006498695321896991;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.141444921493531162) ) ) {
              if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                result[0] += -0.0011935308798214667;
              } else {
                result[0] += -0.057711393977446894;
              }
            } else {
              result[0] += 0.03632188670773461;
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)9.500000000000001776) ) ) {
        if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
          result[0] += -0.08297180838855138;
        } else {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
            result[0] += -0.0038969538307297583;
          } else {
            result[0] += -0.04634358011198616;
          }
        }
      } else {
        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)10.50000000000000178) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.302512168884278232) ) ) {
            result[0] += 0.022008656898342;
          } else {
            result[0] += 0.13064717285931585;
          }
        } else {
          result[0] += -0.04102463700645226;
        }
      }
    }
  }
  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
    if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.597323656082154208) ) ) {
        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
          result[0] += 0.0046583337736690895;
        } else {
          result[0] += -0.025348283809632794;
        }
      } else {
        result[0] += 0.037121700743156245;
      }
    } else {
      if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.284418344497681552) ) ) {
          result[0] += -0.006314919471653673;
        } else {
          if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)47227863040.00000763) ) ) {
            result[0] += -0.016914413566761693;
          } else {
            result[0] += -0.0779575451917871;
          }
        }
      } else {
        if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.552972793579102007) ) ) {
          result[0] += -0.1034325120579231;
        } else {
          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
            result[0] += 0.00032742052926182423;
          } else {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.029068946838379794) ) ) {
              if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                  result[0] += -0.10393650135256559;
                } else {
                  if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                    result[0] += 0.05918629681419624;
                  } else {
                    result[0] += -0.0035624109536756795;
                  }
                }
              } else {
                result[0] += -0.07580764569600383;
              }
            } else {
              result[0] += 0.03619401305751783;
            }
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)137422176256.0000153) ) ) {
      if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.19876670837402521) ) ) {
          if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)8816427008.000001907) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.607751369476319248) ) ) {
              result[0] += 0.08882431801318304;
            } else {
              result[0] += -0.04693296420129922;
            }
          } else {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += -0.007317713174048456;
            } else {
              if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += 0.13701461619557564;
              } else {
                result[0] += 0.013230006174740497;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)7.500000000000000888) ) ) {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.909855604171753818) ) ) {
                if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)21466447872.00000381) ) ) {
                  result[0] += -1.202982028586481e-05;
                } else {
                  result[0] += 0.0181037227811564;
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.172047138214112216) ) ) {
                  if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                    if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += -0.08893308839658413;
                    } else {
                      result[0] += 0.02104414882509474;
                    }
                  } else {
                    result[0] += 0.03812424708842989;
                  }
                } else {
                  result[0] += -0.05571883759047944;
                }
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.284418344497681552) ) ) {
                result[0] += -0.05660672548537349;
              } else {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.909855604171753818) ) ) {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
                      if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                        if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += 0.0709667687713341;
                        } else {
                          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.329314231872559482) ) ) {
                            if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                              result[0] += -0.042803021616636835;
                            } else {
                              result[0] += -0.10642264582408634;
                            }
                          } else {
                            result[0] += 0.003213089186485852;
                          }
                        }
                      } else {
                        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                          result[0] += 0.04062530277649799;
                        } else {
                          result[0] += -0.09436103925023229;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.397998809814454013) ) ) {
                          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                            result[0] += -0.12176531251507992;
                          } else {
                            result[0] += -0.0604591683589678;
                          }
                        } else {
                          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.516936540603638583) ) ) {
                            result[0] += 0.06614255042434018;
                          } else {
                            result[0] += -0.032044118885720944;
                          }
                        }
                      } else {
                        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                          result[0] += 0.0007827589470560472;
                        } else {
                          result[0] += -0.1430153063572501;
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.484580039978028232) ) ) {
                        if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                          result[0] += -0.010527209955554171;
                        } else {
                          result[0] += -0.18648020331678278;
                        }
                      } else {
                        if ( UNLIKELY(  (data[43].missing != -1) && (data[43].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                          if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.029068946838379794) ) ) {
                              result[0] += 0.053589674533288945;
                            } else {
                              result[0] += -0.08400999103696752;
                            }
                          } else {
                            if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
                              result[0] += 0.015370872147764635;
                            } else {
                              result[0] += 0.06756832200638556;
                            }
                          }
                        } else {
                          result[0] += -0.07354038660451838;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
                        result[0] += -0.1713862542230139;
                      } else {
                        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                          result[0] += 0.09120829203983019;
                        } else {
                          result[0] += -0.09200954155146313;
                        }
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += -0.025346794396291686;
                    } else {
                      result[0] += 0.08807290846952387;
                    }
                  } else {
                    result[0] += 0.03544878511457986;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += -0.05324426172813053;
            } else {
              if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                  result[0] += -0.015492740801394803;
                } else {
                  result[0] += 0.11213814813928627;
                }
              } else {
                result[0] += -0.03218774997297989;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
          if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
            result[0] += -0.028544911319044894;
          } else {
            result[0] += 0.10952525761410258;
          }
        } else {
          result[0] += 0.008037973005588754;
        }
      }
    } else {
      if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)9.500000000000001776) ) ) {
        if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
          result[0] += -0.08015810183742783;
        } else {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
            result[0] += -0.005463739968290722;
          } else {
            result[0] += -0.0438381843612227;
          }
        }
      } else {
        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.861792564392090288) ) ) {
          result[0] += -0.04610770038341437;
        } else {
          if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
            result[0] += -0.018204253284955594;
          } else {
            result[0] += 0.06174304338256234;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)2.087193608283997026) ) ) {
      if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)10.50000000000000178) ) ) {
        result[0] += 0.0030650728864964564;
      } else {
        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)46.00000000000000711) ) ) {
          result[0] += 0.05574158849179121;
        } else {
          result[0] += -0.047945745956013724;
        }
      }
    } else {
      result[0] += 0.053135920566546924;
    }
  } else {
    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.172047138214112216) ) ) {
        if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)8816427008.000001907) ) ) {
              result[0] += 0.05934772594816265;
            } else {
              result[0] += -0.01846481308376516;
            }
          } else {
            result[0] += -0.0006155385816073583;
          }
        } else {
          result[0] += 0.013936013645196282;
        }
      } else {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.718933820724488193) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
            result[0] += -0.0016985638929266481;
          } else {
            result[0] += -0.027943061082160704;
          }
        } else {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)46.00000000000000711) ) ) {
            result[0] += 0.0166247219238807;
          } else {
            result[0] += -0.045009208281857074;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.172047138214112216) ) ) {
        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
            if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.827801465988160068) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.87548160552978693) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.208071470260621005) ) ) {
                    result[0] += -0.0003318040811554273;
                  } else {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.329314231872559482) ) ) {
                      if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += -0.0354409024381899;
                      } else {
                        result[0] += -0.13418695297485214;
                      }
                    } else {
                      result[0] += 0.0058045279271629596;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += 0.053090871819694456;
                  } else {
                    result[0] += -0.05854268548510985;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                  if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += -0.09879307692263896;
                  } else {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                      result[0] += -0.05953412346590747;
                    } else {
                      result[0] += 0.01936969420586317;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
                    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                      result[0] += 0.09327508349164709;
                    } else {
                      if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += 0.03780486635138283;
                      } else {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.827801465988160068) ) ) {
                          result[0] += 0.028251616577815965;
                        } else {
                          result[0] += -0.13225587496539964;
                        }
                      }
                    }
                  } else {
                    result[0] += -0.04605721628757008;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.802901029586792436) ) ) {
                result[0] += 0.005249944013958355;
              } else {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
                  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)6.000000000000000888) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
                      result[0] += -0.024110373361616873;
                    } else {
                      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                        result[0] += 0.145230056869517;
                      } else {
                        result[0] += -0.03674833586705118;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.006017134171686804;
                    } else {
                      result[0] += -0.0732919279956793;
                    }
                  }
                } else {
                  result[0] += -0.009212862756986896;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
              result[0] += -0.017330258007832965;
            } else {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.743881702423096591) ) ) {
                  result[0] += 0.049656405102822036;
                } else {
                  result[0] += -0.06756305072148852;
                }
              } else {
                result[0] += 0.003876125723554201;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
            result[0] += -0.024069679846458924;
          } else {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
              result[0] += -0.08613819753622087;
            } else {
              result[0] += 0.042291262967885594;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)40.00000000000000711) ) ) {
          result[0] += -0.05283619178153962;
        } else {
          if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.08214424513233486;
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.607751369476319248) ) ) {
                  if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                    result[0] += -0.08489938710742481;
                  } else {
                    result[0] += -0.012324977104226297;
                  }
                } else {
                  if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
                    result[0] += -0.016779770757249186;
                  } else {
                    if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.161602735519410068) ) ) {
                        result[0] += -0.013922973648115108;
                      } else {
                        if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                          result[0] += -0.009650731458083416;
                        } else {
                          result[0] += 0.08366127845536361;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                          result[0] += 0.06669419028597377;
                        } else {
                          result[0] += -0.06917741410760499;
                        }
                      } else {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.070054531097412998) ) ) {
                          result[0] += -0.016134517950659965;
                        } else {
                          result[0] += 0.07643923871232128;
                        }
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                  result[0] += -0.027694540246851192;
                } else {
                  result[0] += 0.031111845369667615;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.022130427305008086;
                } else {
                  result[0] += 0.01820384772856972;
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.484580039978028232) ) ) {
                  if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                    result[0] += 0.023903193925666526;
                  } else {
                    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                      result[0] += -0.1327925959390919;
                    } else {
                      result[0] += 0.0021646345899907424;
                    }
                  }
                } else {
                  result[0] += 0.03705483511806728;
                }
              }
            } else {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                  result[0] += 0.03406170551913942;
                } else {
                  result[0] += -0.08475826509972775;
                }
              } else {
                result[0] += 0.0258675039612496;
              }
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
    if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)24.00000000000000355) ) ) {
      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
        if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += 0.004705768675590904;
        } else {
          if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
                if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.0663375122722276;
                } else {
                  result[0] += -0.010587406028100647;
                }
              } else {
                result[0] += -0.021168338203115186;
              }
            } else {
              if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += -0.00833117112899537;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.172047138214112216) ) ) {
                  if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += -0.05167413701300627;
                  } else {
                    result[0] += 0.010558769784289273;
                  }
                } else {
                  result[0] += 0.0252923631905633;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.142630577087403232) ) ) {
              if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)14.74696540832519709) ) ) {
                  result[0] += 0.026548809179597416;
                } else {
                  result[0] += 0.09928390442408014;
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.154959201812744585) ) ) {
                  result[0] += -0.030583117556694323;
                } else {
                  result[0] += 0.014223618161798058;
                }
              }
            } else {
              if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.651049375534058505) ) ) {
                  result[0] += -0.0053192820073333326;
                } else {
                  result[0] += -0.035251347425963;
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.284418344497681552) ) ) {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += -0.01698389399041794;
                  } else {
                    result[0] += 0.034189719825148526;
                  }
                } else {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                    result[0] += 0.00693410363487582;
                  } else {
                    result[0] += 0.03639044911261099;
                  }
                }
              }
            }
          }
        }
      } else {
        result[0] += -0.023024734952828182;
      }
    } else {
      result[0] += -0.10732427015191603;
    }
  } else {
    if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.48738741874694913) ) ) {
          if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2415.000000000000455) ) ) {
            result[0] += 0.0018052594603417207;
          } else {
            result[0] += 0.019887464228916425;
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.172047138214112216) ) ) {
            if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
              result[0] += -0.041517373227692556;
            } else {
              result[0] += 0.043021714506552455;
            }
          } else {
            if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)3072.000000000000455) ) ) {
              result[0] += -0.008910669289923316;
            } else {
              result[0] += -0.06462528526275291;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.484580039978028232) ) ) {
          if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
            if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
              result[0] += -0.0009460969282451189;
            } else {
              result[0] += -0.019515536850621525;
            }
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.357691764831543413) ) ) {
              result[0] += 0.09921378082776591;
            } else {
              result[0] += -0.010505026043064455;
            }
          }
        } else {
          if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
            result[0] += -0.07941753162467036;
          } else {
            if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += 0.009395876157095516;
            } else {
              result[0] += -0.029862371278389994;
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)6.000000000000000888) ) ) {
            if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
              result[0] += -0.08083244385466522;
            } else {
              if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                  result[0] += 0.03043298345676304;
                } else {
                  result[0] += -0.031467980660143224;
                }
              } else {
                result[0] += -0.08692594930770388;
              }
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.172047138214112216) ) ) {
              if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.029068946838379794) ) ) {
                  result[0] += 0.01659138386018575;
                } else {
                  result[0] += -0.08649799236492516;
                }
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.255632162094117099) ) ) {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.993164777755738193) ) ) {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.909855604171753818) ) ) {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.02604460716247603) ) ) {
                          result[0] += 0.029150284985798736;
                        } else {
                          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
                            result[0] += -0.10532393126416001;
                          } else {
                            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.802901029586792436) ) ) {
                              result[0] += 0.030456505380284344;
                            } else {
                              result[0] += -0.0447197341074507;
                            }
                          }
                        }
                      } else {
                        result[0] += 0.05690174951458347;
                      }
                    } else {
                      result[0] += -0.09589769991617204;
                    }
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.651049375534058505) ) ) {
                      result[0] += -0.01831459261228354;
                    } else {
                      result[0] += 0.08860922409080724;
                    }
                  }
                } else {
                  result[0] += 0.00502720599233276;
                }
              }
            } else {
              result[0] += 0.007616275692622729;
            }
          }
        } else {
          if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
            result[0] += -0.0843441462400873;
          } else {
            result[0] += -0.007329434324012495;
          }
        }
      } else {
        if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)2.500000000000000444) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
            if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.64763975143432706) ) ) {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)15.26177501678466975) ) ) {
                  result[0] += 0.010160371694281535;
                } else {
                  result[0] += -0.07417804142800179;
                }
              } else {
                result[0] += 0.12237251494976209;
              }
            } else {
              result[0] += -0.07517612951917889;
            }
          } else {
            if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)8.500000000000001776) ) ) {
              result[0] += -0.025226795563688953;
            } else {
              result[0] += 0.08456409012502905;
            }
          }
        } else {
          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += -0.09259600590600964;
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.802696108818054643) ) ) {
                if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.06922562326608543;
                } else {
                  result[0] += -0.030571965738953977;
                }
              } else {
                if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)10.50000000000000178) ) ) {
                  result[0] += 0.043333374156456786;
                } else {
                  result[0] += -0.014208594385444593;
                }
              }
            }
          } else {
            result[0] += -0.009049638193056092;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)3.000000000000000444) ) ) {
      if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)257681260544.0000305) ) ) {
        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
          if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.760117292404175249) ) ) {
            if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.764703154563904253) ) ) {
                result[0] += 0.003160826234849392;
              } else {
                result[0] += 0.020997849177686302;
              }
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.357691764831543413) ) ) {
                result[0] += 0.05073071978137403;
              } else {
                result[0] += -0.000504317881869649;
              }
            }
          } else {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += 0.021855844367157654;
            } else {
              if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.284418344497681552) ) ) {
                result[0] += 0.1275432614301098;
              } else {
                result[0] += -0.03183884820397148;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.24492526054382413) ) ) {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
                result[0] += 0.02338489238891461;
              } else {
                if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                  if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.827801465988160068) ) ) {
                      result[0] += -0.004150056366347483;
                    } else {
                      result[0] += -0.07338872133418738;
                    }
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                      result[0] += -0.10693952624001449;
                    } else {
                      result[0] += 0.03166458232477036;
                    }
                  }
                } else {
                  result[0] += 0.07265822437594575;
                }
              }
            } else {
              result[0] += 0.0806156797226334;
            }
          } else {
            result[0] += 7.164742443573554e-05;
          }
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.215408444404602495) ) ) {
          result[0] += 0.016522995425949247;
        } else {
          result[0] += 0.06566116760209283;
        }
      }
    } else {
      if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
          result[0] += -0.00493502633940223;
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.154959201812744585) ) ) {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)40.00000000000000711) ) ) {
              result[0] += 0.044588436213690616;
            } else {
              result[0] += -0.037181004839074425;
            }
          } else {
            result[0] += -0.02707237116386967;
          }
        }
      } else {
        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.088880300521851474) ) ) {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.484580039978028232) ) ) {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.014892877286637712;
              } else {
                result[0] += 0.08403400632755112;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.599987030029298651) ) ) {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.607751369476319248) ) ) {
                    result[0] += -0.043351380868674266;
                  } else {
                    result[0] += 0.027474600646170633;
                  }
                } else {
                  result[0] += 0.039698400862325525;
                }
              } else {
                if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
                      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                        result[0] += -0.034287913086031775;
                      } else {
                        result[0] += 0.0022273402033460843;
                      }
                    } else {
                      result[0] += 0.003155224943312891;
                    }
                  } else {
                    if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                      result[0] += -0.08275298723087161;
                    } else {
                      result[0] += 0.01649478731206054;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                    if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.021028568144442074;
                    } else {
                      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                        if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.870983839035034624) ) ) {
                          result[0] += -0.013280285186866043;
                        } else {
                          result[0] += 0.10384186815444088;
                        }
                      } else {
                        result[0] += -0.014320641254330333;
                      }
                    }
                  } else {
                    result[0] += -0.03671523750200839;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.484580039978028232) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.215408444404602495) ) ) {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.764703154563904253) ) ) {
                  if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)8816427008.000001907) ) ) {
                    result[0] += 0.13549469664012015;
                  } else {
                    result[0] += 0.014466420059247956;
                  }
                } else {
                  result[0] += -0.021925788330367874;
                }
              } else {
                result[0] += -0.05151905901752654;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.481347560882569248) ) ) {
                if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.002403334084556129;
                  } else {
                    result[0] += -0.1073758977348237;
                  }
                } else {
                  result[0] += -0.11291375838440142;
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.736135363578796831) ) ) {
                  result[0] += 0.17319217170414936;
                } else {
                  if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                    if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                        result[0] += 0.0409939086101311;
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.055836200714113104) ) ) {
                          result[0] += -0.06836296693396007;
                        } else {
                          result[0] += 0.0025677189774051747;
                        }
                      }
                    } else {
                      result[0] += -0.01115642206426102;
                    }
                  } else {
                    result[0] += 0.014492678125398187;
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
                if ( UNLIKELY(  (data[44].missing != -1) && (data[44].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                  result[0] += -0.013546187629806104;
                } else {
                  result[0] += 0.03418711121866926;
                }
              } else {
                result[0] += -0.03060589258633957;
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
                result[0] += -0.05766824490884009;
              } else {
                if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)4.500000000000000888) ) ) {
                  if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += 0.00303661444006817;
                  } else {
                    result[0] += 0.10998244936539087;
                  }
                } else {
                  result[0] += -0.030326797679613104;
                }
              }
            }
          } else {
            result[0] += 0.022949687588503128;
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
      if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)47227863040.00000763) ) ) {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.233438730239869052) ) ) {
          if ( LIKELY( !(data[10].missing != -1) || (data[10].fvalue <= (double)0.8958797454833985485) ) ) {
            result[0] += -0.0004185755016479687;
          } else {
            result[0] += -0.09588910787436525;
          }
        } else {
          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
            result[0] += -0.02417508637044907;
          } else {
            result[0] += 0.04426726987101824;
          }
        }
      } else {
        result[0] += -0.04545503826522543;
      }
    } else {
      result[0] += 0.00826638531149763;
    }
  }
  if ( UNLIKELY(  (data[45].missing != -1) && (data[45].fvalue <= (double)-1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)257681260544.0000305) ) ) {
      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
          result[0] += 0.05208410752020416;
        } else {
          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.888826131820679155) ) ) {
              result[0] += 0.008409493475923714;
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.036049604415894443) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.051912069320679599) ) ) {
                  result[0] += 0.06010351368494185;
                } else {
                  result[0] += 0.015346971042057482;
                }
              } else {
                result[0] += 0.11874651113045642;
              }
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.516936540603638583) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.42478513717651456) ) ) {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += 0.006823376410451634;
                } else {
                  result[0] += -0.01318847289414199;
                }
              } else {
                result[0] += -0.030721025304866163;
              }
            } else {
              result[0] += 0.018451568948290184;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.311204195022583896) ) ) {
            result[0] += -0.040056619450238196;
          } else {
            result[0] += 0.08376336658462934;
          }
        } else {
          result[0] += -6.21926872304383e-05;
        }
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.215408444404602495) ) ) {
        result[0] += 0.015585689024071488;
      } else {
        result[0] += 0.06303994901238923;
      }
    }
  } else {
    if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
          if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)2.500000000000000444) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.142630577087403232) ) ) {
              if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.07729864120483576) ) ) {
                  if ( UNLIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)8816427008.000001907) ) ) {
                      result[0] += -0.13606358745411637;
                    } else {
                      result[0] += -0.008021963451779529;
                    }
                  } else {
                    if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)8816427008.000001907) ) ) {
                      result[0] += 0.18232608486104154;
                    } else {
                      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                        result[0] += 0.015086646992100224;
                      } else {
                        result[0] += -0.12997391531553926;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.56849193572998225) ) ) {
                    result[0] += 0.04142967477956553;
                  } else {
                    if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += -0.12821295681441522;
                    } else {
                      result[0] += -0.0007613499767779906;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)8816427008.000001907) ) ) {
                  result[0] += 0.06829941696044643;
                } else {
                  result[0] += -0.020894610739019184;
                }
              }
            } else {
              if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.760117292404175249) ) ) {
                result[0] += -0.037090544241180516;
              } else {
                result[0] += -0.004741113228913966;
              }
            }
          } else {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
              if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.778982400894165927) ) ) {
                  result[0] += -0.004113003978228403;
                } else {
                  if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += -0.023345497419331836;
                  } else {
                    result[0] += -0.07850524048882633;
                  }
                }
              } else {
                if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.731144905090333808) ) ) {
                    result[0] += -0.10267834085532479;
                  } else {
                    result[0] += -0.003683940894661657;
                  }
                } else {
                  result[0] += 0.03240219394116727;
                }
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
                result[0] += -0.024539945894628698;
              } else {
                result[0] += 0.012748688860086189;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.154959201812744585) ) ) {
            if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.02635641640138265;
            } else {
              result[0] += 0.04997226969555907;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.41211462020874201) ) ) {
              if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.07212984501888473;
                } else {
                  if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                    if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.362805843353273261) ) ) {
                      result[0] += -0.039939600019995206;
                    } else {
                      result[0] += 0.012552115394913297;
                    }
                  } else {
                    if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += -0.09678014840334445;
                    } else {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.45023441314697443) ) ) {
                        result[0] += 0.06720884540030755;
                      } else {
                        result[0] += -0.0033929029195810716;
                      }
                    }
                  }
                }
              } else {
                result[0] += 0.025942171889165686;
              }
            } else {
              result[0] += -0.03710909628472448;
            }
          }
        }
      } else {
        result[0] += -0.0011005978533835759;
      }
    } else {
      if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.233438730239869052) ) ) {
            if ( LIKELY( !(data[10].missing != -1) || (data[10].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += -0.00022182554773078202;
            } else {
              result[0] += -0.08458350375004403;
            }
          } else {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += -0.02423979069717143;
            } else {
              if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.070054531097412998) ) ) {
                  result[0] += 0.08781398728598877;
                } else {
                  result[0] += -0.06849087925251633;
                }
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.644374847412110263) ) ) {
                  result[0] += 0.03356527248589736;
                } else {
                  result[0] += -0.09259941612607564;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.21969318389892756) ) ) {
            result[0] += 0.022813701880320725;
          } else {
            result[0] += -0.05388626917892761;
          }
        }
      } else {
        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.94957673549652144) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.357691764831543413) ) ) {
              result[0] += 0.03174226065118365;
            } else {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += 0.004355665254820958;
              } else {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                  result[0] += -0.06012003439195601;
                } else {
                  result[0] += -0.01542724041232868;
                }
              }
            }
          } else {
            result[0] += 0.008180054292942095;
          }
        } else {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.605039834976196733) ) ) {
            if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)6.023992538452149326) ) ) {
              result[0] += 0.04510576119403116;
            } else {
              result[0] += -0.026006463816468397;
            }
          } else {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += -0.01827483657269965;
            } else {
              result[0] += 0.020876934697907495;
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)3.000000000000000444) ) ) {
    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.572941064834595615) ) ) {
          result[0] += 0.006086182175339765;
        } else {
          result[0] += 0.06875208592364926;
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.215408444404602495) ) ) {
          result[0] += 0.015347767039253574;
        } else {
          result[0] += 0.06772141407152847;
        }
      }
    } else {
      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.208071470260621005) ) ) {
        result[0] += -0.0015261665442310149;
      } else {
        result[0] += -0.04935447493332016;
      }
    }
  } else {
    if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
        if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
            if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)257681260544.0000305) ) ) {
              result[0] += -0.011057024486838546;
            } else {
              result[0] += -0.034809845077662785;
            }
          } else {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.801661729812622958) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.114721298217775214) ) ) {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += -0.0013485459376218626;
                  } else {
                    result[0] += 0.03428379618033597;
                  }
                } else {
                  result[0] += 0.0019210038777029365;
                }
              } else {
                result[0] += -0.03234301476185774;
              }
            } else {
              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.552972793579102007) ) ) {
                result[0] += -0.1058136604615479;
              } else {
                if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.11235428395071026;
                    } else {
                      result[0] += 0.023262023228206117;
                    }
                  } else {
                    result[0] += -0.03752304416611566;
                  }
                } else {
                  if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.484580039978028232) ) ) {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
                        result[0] += -0.00012705604800308982;
                      } else {
                        result[0] += -0.0609194899107522;
                      }
                    } else {
                      if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += -0.009224069321950254;
                      } else {
                        result[0] += 0.019017515416470135;
                      }
                    }
                  } else {
                    result[0] += -0.015645482846592895;
                  }
                }
              }
            }
          }
        } else {
          result[0] += 0.005680855874385254;
        }
      } else {
        if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
            result[0] += -0.013398712167501138;
          } else {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
              result[0] += -0.03126630416521575;
            } else {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
                if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.778982400894165927) ) ) {
                        result[0] += 0.004955332724933591;
                      } else {
                        result[0] += -0.04739268758245839;
                      }
                    } else {
                      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.302512168884278232) ) ) {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.484580039978028232) ) ) {
                          result[0] += -0.0559174105505927;
                        } else {
                          result[0] += 0.015642787742313112;
                        }
                      } else {
                        result[0] += 0.07507628542360689;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.607751369476319248) ) ) {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
                        result[0] += 0.06336406764436431;
                      } else {
                        result[0] += -0.00469995682863204;
                      }
                    } else {
                      result[0] += 0.05432278068116347;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.172047138214112216) ) ) {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                      result[0] += 0.0032018984652556585;
                    } else {
                      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.827801465988160068) ) ) {
                        result[0] += -0.034173690145791934;
                      } else {
                        result[0] += -0.11459147132403141;
                      }
                    }
                  } else {
                    result[0] += 0.01709678128149505;
                  }
                }
              } else {
                result[0] += -0.0426241551678866;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
            if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)4.500000000000000888) ) ) {
              result[0] += 0.008319442435441539;
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.467917680740357333) ) ) {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.484580039978028232) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)15.26177501678466975) ) ) {
                      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.694163918495178667) ) ) {
                          result[0] += 0.062412498854293834;
                        } else {
                          result[0] += 0.0012776464781685845;
                        }
                      } else {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.694163918495178667) ) ) {
                          result[0] += -0.052370676858876045;
                        } else {
                          result[0] += 0.055825239105782326;
                        }
                      }
                    } else {
                      result[0] += -0.04065818378088808;
                    }
                  } else {
                    if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.030091470022264935;
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.172047138214112216) ) ) {
                        result[0] += -0.02570663282497811;
                      } else {
                        result[0] += 0.06246691677812504;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.484580039978028232) ) ) {
                    result[0] += -0.07210092429198725;
                  } else {
                    result[0] += 0.00789306117552456;
                  }
                }
              } else {
                if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.0324616600268247;
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.484580039978028232) ) ) {
                    result[0] += 0.3867549066512222;
                  } else {
                    result[0] += 0.010412067630823781;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.484580039978028232) ) ) {
              result[0] += -0.027339608998923622;
            } else {
              if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.201062679290773261) ) ) {
                  result[0] += -0.11325802174946722;
                } else {
                  result[0] += 0.05265822208903934;
                }
              } else {
                if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += -0.05650818443121375;
                } else {
                  if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.06471203346228635;
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.302512168884278232) ) ) {
                      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                        result[0] += -0.16426847396886368;
                      } else {
                        result[0] += 0.034564212653290106;
                      }
                    } else {
                      result[0] += 0.02050240022016295;
                    }
                  }
                }
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
        result[0] += 0.0035612186931923413;
      } else {
        if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.743881702423096591) ) ) {
            result[0] += -0.016613686019579983;
          } else {
            result[0] += 0.09932540732376888;
          }
        } else {
          result[0] += -0.06551386434493826;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)3.000000000000000444) ) ) {
    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
      if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2252.000000000000455) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.215408444404602495) ) ) {
          result[0] += 0.014598788373417715;
        } else {
          result[0] += 0.06511810714473178;
        }
      } else {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.572941064834595615) ) ) {
          if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
            if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.760117292404175249) ) ) {
              if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.764703154563904253) ) ) {
                  result[0] += 0.0031334592784625797;
                } else {
                  result[0] += 0.01873787678165087;
                }
              } else {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.42478513717651456) ) ) {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += 0.006832130586896446;
                  } else {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.827801465988160068) ) ) {
                      result[0] += -0.019159772370408917;
                    } else {
                      result[0] += 0.009839986054059224;
                    }
                  }
                } else {
                  result[0] += -0.023056961658442662;
                }
              }
            } else {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += 0.019596164982670458;
              } else {
                if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.201062679290773261) ) ) {
                  result[0] += 0.11242544037843467;
                } else {
                  result[0] += -0.028445234208461602;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.802696108818054643) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.04636812210083185) ) ) {
                result[0] += 0.02376572720088465;
              } else {
                result[0] += -0.055624160393736945;
              }
            } else {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.90474271774292081) ) ) {
                  result[0] += -0.07446883493930226;
                } else {
                  result[0] += 0.05182314655473446;
                }
              } else {
                result[0] += 0.057060600126864584;
              }
            }
          }
        } else {
          result[0] += 0.06578842676759097;
        }
      }
    } else {
      result[0] += -0.0023740332132228498;
    }
  } else {
    if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.802696108818054643) ) ) {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.45958471298217951) ) ) {
              result[0] += 0.002983334715830323;
            } else {
              result[0] += 0.14121203521844183;
            }
          } else {
            if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)8816427008.000001907) ) ) {
              if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.0013250801627438785;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.04969787597656428) ) ) {
                  result[0] += 0.14924730332792813;
                } else {
                  result[0] += 0.0013440796520342137;
                }
              }
            } else {
              result[0] += -0.007346759660511484;
            }
          }
        } else {
          if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
            result[0] += -0.08215974461711112;
          } else {
            result[0] += -0.02245214955692412;
          }
        }
      } else {
        if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
          if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.825422286987305576) ) ) {
              if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.135017871856690341) ) ) {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += 0.0025429611337467544;
                  } else {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                      result[0] += 0.07665687509677949;
                    } else {
                      result[0] += -0.008449247069831741;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.357691764831543413) ) ) {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
                        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                          result[0] += 0.1309868219876806;
                        } else {
                          result[0] += -0.029757206267979164;
                        }
                      } else {
                        result[0] += 0.04819066383464687;
                      }
                    } else {
                      result[0] += -0.0326619069267655;
                    }
                  } else {
                    if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.021724001679346076;
                    } else {
                      if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                        if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += -0.035661986568753135;
                        } else {
                          result[0] += 0.00913625319794621;
                        }
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.481347560882569248) ) ) {
                          result[0] += -0.11351876613063872;
                        } else {
                          if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                            result[0] += 0.02555881306148779;
                          } else {
                            result[0] += -0.002363104173042465;
                          }
                        }
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                  result[0] += -0.008973504188687474;
                } else {
                  if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.142630577087403232) ) ) {
                      result[0] += 0.08731665133703641;
                    } else {
                      result[0] += -0.0018034466088668773;
                    }
                  } else {
                    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.484580039978028232) ) ) {
                        result[0] += -0.03734807685454571;
                      } else {
                        result[0] += 0.08188531200317914;
                      }
                    } else {
                      result[0] += 0.019293959751593356;
                    }
                  }
                }
              }
            } else {
              result[0] += -0.03566814413758658;
            }
          } else {
            if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
              if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += -0.02112380861814642;
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.302512168884278232) ) ) {
                  result[0] += -0.004611457403369848;
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.607751369476319248) ) ) {
                    result[0] += 0.06554494839634299;
                  } else {
                    result[0] += 0.010867373040591856;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += 0.023383945405495737;
              } else {
                result[0] += -0.028202112102734763;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.552972793579102007) ) ) {
            result[0] += -0.09077665350394253;
          } else {
            result[0] += 0.0005359729775023542;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
        if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)47227863040.00000763) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.261864185333252841) ) ) {
            if ( LIKELY( !(data[10].missing != -1) || (data[10].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += -0.0008768881753502272;
            } else {
              result[0] += -0.08685597405637702;
            }
          } else {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += -0.021819235187826812;
            } else {
              if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.070054531097412998) ) ) {
                  result[0] += 0.08600383832956343;
                } else {
                  result[0] += -0.06422108762410192;
                }
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.644374847412110263) ) ) {
                  result[0] += 0.03042576416131776;
                } else {
                  result[0] += -0.0922203940000006;
                }
              }
            }
          }
        } else {
          result[0] += -0.04282326818223565;
        }
      } else {
        result[0] += 0.0063009803999081295;
      }
    }
  }
  if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)3.000000000000000444) ) ) {
    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.572941064834595615) ) ) {
      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2252.000000000000455) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.135017871856690341) ) ) {
            result[0] += -0.04965314790907852;
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.215408444404602495) ) ) {
              result[0] += 0.0164759684519978;
            } else {
              result[0] += 0.06766585605283738;
            }
          }
        } else {
          result[0] += 0.005394220429388852;
        }
      } else {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.607751369476319248) ) ) {
          result[0] += -0.0005634454974882108;
        } else {
          result[0] += -0.03794461047706407;
        }
      }
    } else {
      result[0] += 0.05542509280922428;
    }
  } else {
    if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.802696108818054643) ) ) {
          if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)8816427008.000001907) ) ) {
            if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += -0.026045570460496603;
              } else {
                result[0] += 0.07092626422145379;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.04969787597656428) ) ) {
                result[0] += 0.16405124996119008;
              } else {
                result[0] += -0.006969855670051036;
              }
            }
          } else {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.03841849430092302;
                } else {
                  result[0] += -0.05953818916895631;
                }
              } else {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.764703154563904253) ) ) {
                  result[0] += 0.005247964083069033;
                } else {
                  result[0] += -0.15393977420510363;
                }
              }
            } else {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.45958471298217951) ) ) {
                  result[0] += 0.0033839973352378784;
                } else {
                  result[0] += 0.13357485629345728;
                }
              } else {
                if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
                  if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.142630577087403232) ) ) {
                      result[0] += -0.007000014548912064;
                    } else {
                      result[0] += 0.0446559577931535;
                    }
                  } else {
                    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.06492702416114259;
                    } else {
                      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.671254158020020419) ) ) {
                        result[0] += -0.06930032612620184;
                      } else {
                        result[0] += 0.03835950239845091;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.23969554901123225) ) ) {
                    result[0] += -0.006911223487120631;
                  } else {
                    result[0] += -0.03149342710973151;
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
            result[0] += -0.07963560322414642;
          } else {
            result[0] += -0.021450681478293283;
          }
        }
      } else {
        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)9.500000000000001776) ) ) {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.623839378356934482) ) ) {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.909855604171753818) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.484580039978028232) ) ) {
                  result[0] += 0.009740599460864339;
                } else {
                  result[0] += -0.0022231687747769455;
                }
              } else {
                result[0] += -0.03309420271953177;
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.484580039978028232) ) ) {
                result[0] += -0.05070630248687097;
              } else {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                  result[0] += -0.004610048589922441;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.894675970077515537) ) ) {
                    result[0] += -0.10514159984294767;
                  } else {
                    if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.467917680740357333) ) ) {
                      result[0] += 0.0073745197628187365;
                    } else {
                      result[0] += 0.033842882022073355;
                    }
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += -0.041004124775157795;
            } else {
              if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += -0.053186597286285844;
                } else {
                  result[0] += 0.048768200293014284;
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.029068946838379794) ) ) {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)208.0000000000000284) ) ) {
                    result[0] += 0.05599129222575155;
                  } else {
                    result[0] += -0.06532340060961028;
                  }
                } else {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.516936540603638583) ) ) {
                      result[0] += -0.05421897141921503;
                    } else {
                      result[0] += 0.008527080480334005;
                    }
                  } else {
                    if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)4.500000000000000888) ) ) {
                      if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                        result[0] += 0.11539702147578751;
                      } else {
                        result[0] += 0.03634522583038846;
                      }
                    } else {
                      result[0] += -0.055172492606763636;
                    }
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)10.50000000000000178) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.978769779205324042) ) ) {
              result[0] += -0.08526703362765264;
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.357691764831543413) ) ) {
                result[0] += -0.07988169482480378;
              } else {
                if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.607751369476319248) ) ) {
                    result[0] += 0.04500296582502034;
                  } else {
                    result[0] += 0.13005391939197694;
                  }
                } else {
                  result[0] += 0.030048291583930537;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
              result[0] += 0.06793347985091519;
            } else {
              result[0] += -0.036900239964269825;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.215408444404602495) ) ) {
        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.911822915077209917) ) ) {
          result[0] += 0.005323867112697473;
        } else {
          result[0] += -0.04218199869961218;
        }
      } else {
        if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
          if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.261864185333252841) ) ) {
              result[0] += -0.0005023887012592099;
            } else {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += -0.019969363899205294;
              } else {
                if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.070054531097412998) ) ) {
                    result[0] += 0.07717481959557686;
                  } else {
                    result[0] += -0.0623049567989345;
                  }
                } else {
                  if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                    result[0] += 0.04128917792704708;
                  } else {
                    if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.04259947873061242;
                    } else {
                      result[0] += -0.11280928965559178;
                    }
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.350240230560303178) ) ) {
              result[0] += 0.017040965052641684;
            } else {
              result[0] += -0.0531956863580259;
            }
          }
        } else {
          result[0] += 0.005877541962255944;
        }
      }
    }
  }
  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)10.50000000000000178) ) ) {
    if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)9.500000000000001776) ) ) {
      if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
        result[0] += 0.0004942009279669573;
      } else {
        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
          if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
            if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
              if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.497866153717041238) ) ) {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
                      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.255632162094117099) ) ) {
                        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.285887241363526279) ) ) {
                          if ( LIKELY(  (data[42].missing != -1) && (data[42].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                            result[0] += 0.029685651914028652;
                          } else {
                            result[0] += -0.008543405989809844;
                          }
                        } else {
                          result[0] += -0.05926187581409096;
                        }
                      } else {
                        result[0] += -0.0490915554233945;
                      }
                    } else {
                      result[0] += -0.03799307419124189;
                    }
                  } else {
                    result[0] += 0.10821961047760527;
                  }
                } else {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)46.00000000000000711) ) ) {
                    if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.64270687103271662) ) ) {
                        if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                          result[0] += -0.11173393034776068;
                        } else {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.41211462020874201) ) ) {
                            result[0] += 0.04762919031043604;
                          } else {
                            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.516936540603638583) ) ) {
                              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                                if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                                  result[0] += 0.06934382380159225;
                                } else {
                                  result[0] += -0.03696505862032792;
                                }
                              } else {
                                if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.827801465988160068) ) ) {
                                    result[0] += -0.1609589162613896;
                                  } else {
                                    result[0] += -0.030034054167335372;
                                  }
                                } else {
                                  result[0] += 0.020835264829211338;
                                }
                              }
                            } else {
                              result[0] += -0.0994289716945244;
                            }
                          }
                        }
                      } else {
                        result[0] += 0.09382438739977471;
                      }
                    } else {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.651049375534058505) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.888949394226075995) ) ) {
                          result[0] += 0.04171864881087708;
                        } else {
                          result[0] += -0.013908648969436575;
                        }
                      } else {
                        if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                          result[0] += -0.028497448652380786;
                        } else {
                          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.161602735519410068) ) ) {
                            result[0] += 0.053911309590797575;
                          } else {
                            result[0] += 0.0032681358849335727;
                          }
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.484580039978028232) ) ) {
                        result[0] += 0.09536237237283682;
                      } else {
                        result[0] += 0.015766667698041786;
                      }
                    } else {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += -0.07211222789245754;
                      } else {
                        result[0] += -0.001526257172501785;
                      }
                    }
                  }
                }
              } else {
                result[0] += -0.0542999370829502;
              }
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.849175214767456943) ) ) {
                result[0] += 0.023974173224248933;
              } else {
                result[0] += -0.02188898023107327;
              }
            }
          } else {
            if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
              if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.94957673549652144) ) ) {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                    result[0] += -0.03877575496105693;
                  } else {
                    result[0] += 0.05262239301965663;
                  }
                } else {
                  result[0] += 0.018911722664377587;
                }
              } else {
                if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.05375929031065263;
                } else {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += 0.09867071619135784;
                  } else {
                    if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)6.000000000000000888) ) ) {
                      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                        result[0] += -0.055849706269511426;
                      } else {
                        if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.695749998092652255) ) ) {
                            if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
                              result[0] += 0.04393385099130052;
                            } else {
                              result[0] += 0.3841605026937613;
                            }
                          } else {
                            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.88435244560241788) ) ) {
                              result[0] += 0.032009139920020455;
                            } else {
                              result[0] += 1.2217469450890892;
                            }
                          }
                        } else {
                          result[0] += -0.08811709137429267;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.849175214767456943) ) ) {
                        result[0] += -0.00793628357455963;
                      } else {
                        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                          result[0] += -0.056765283776358824;
                        } else {
                          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.484580039978028232) ) ) {
                            result[0] += -0.12617835600027236;
                          } else {
                            result[0] += 0.04859857064103171;
                          }
                        }
                      }
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.141444921493531162) ) ) {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.014767080095647514;
                    } else {
                      result[0] += 0.21064857120799438;
                    }
                  } else {
                    result[0] += -0.05586021240757212;
                  }
                } else {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += -0.04358023501642741;
                  } else {
                    result[0] += 0.026883751826945475;
                  }
                }
              } else {
                result[0] += -0.059456460290703776;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += -0.07548007794134942;
          } else {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
              result[0] += -0.011595369643258344;
            } else {
              result[0] += -0.04402266978808597;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)257681260544.0000305) ) ) {
        if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.95906782150268732) ) ) {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.0890246292664721;
            } else {
              result[0] += 0.15088206458571968;
            }
          } else {
            result[0] += -0.023700040967778592;
          }
        } else {
          result[0] += 0.08418835070721463;
        }
      } else {
        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.861792564392090288) ) ) {
          result[0] += -0.0482511820014477;
        } else {
          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.361115694046021396) ) ) {
              result[0] += 0.017783712335549037;
            } else {
              result[0] += -0.09398493279251613;
            }
          } else {
            if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += -0.020830149541617528;
            } else {
              result[0] += 0.07333252825249349;
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)46.00000000000000711) ) ) {
      result[0] += 0.048519323954227644;
    } else {
      result[0] += -0.044113581070347546;
    }
  }
  if ( LIKELY(  (data[42].missing != -1) && (data[42].fvalue <= (double)-1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)2.087193608283997026) ) ) {
      if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
        if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
          if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.870983839035034624) ) ) {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.802901029586792436) ) ) {
                result[0] += -0.010171510939849216;
              } else {
                result[0] += 0.008297747605695056;
              }
            } else {
              if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)2.500000000000000444) ) ) {
                result[0] += 0.0192514738885933;
              } else {
                result[0] += 0.07837775252692165;
              }
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.172047138214112216) ) ) {
              result[0] += 0.09373846353126897;
            } else {
              result[0] += -0.02514938854828628;
            }
          }
        } else {
          if ( UNLIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.01053571701049982) ) ) {
                result[0] += 0.026921177946036462;
              } else {
                result[0] += -0.027894232677066078;
              }
            } else {
              result[0] += -0.06722965372155307;
            }
          } else {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.172047138214112216) ) ) {
                if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += 0.043451549889252;
                } else {
                  result[0] += -0.036585825058967646;
                }
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.827801465988160068) ) ) {
                  result[0] += 0.051101867123609326;
                } else {
                  result[0] += -0.07952836573376486;
                }
              }
            } else {
              result[0] += 0.009650077708815284;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)0.8958797454833985485) ) ) {
          result[0] += -0.05329699053260065;
        } else {
          if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
                result[0] += -0.001942476145001388;
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.284418344497681552) ) ) {
                  result[0] += -0.06610966315419835;
                } else {
                  if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)257681260544.0000305) ) ) {
                        result[0] += -0.033852499248947975;
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.029068946838379794) ) ) {
                          result[0] += -0.009754628216567236;
                        } else {
                          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.329314231872559482) ) ) {
                            result[0] += -0.14541974342581765;
                          } else {
                            result[0] += -0.33585285806519005;
                          }
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                        result[0] += -0.02117524753011631;
                      } else {
                        result[0] += 0.04376182232727192;
                      }
                    }
                  } else {
                    result[0] += 0.00808568456019277;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += 0.008123920734719988;
                  } else {
                    result[0] += 0.045641979597685456;
                  }
                } else {
                  result[0] += -0.010414953429590064;
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.599987030029298651) ) ) {
                  if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += 0.00018713555772198218;
                  } else {
                    result[0] += 0.07985029053070536;
                  }
                } else {
                  if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.02778905882759987;
                  } else {
                    if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                      result[0] += 0.016783268811181914;
                    } else {
                      result[0] += -0.01567521758463122;
                    }
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                  if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                    result[0] += 0.08357755222970525;
                  } else {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.87008237838745206) ) ) {
                        if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)7.500000000000000888) ) ) {
                          result[0] += 0.01946492086894153;
                        } else {
                          result[0] += -0.03951207265917547;
                        }
                      } else {
                        result[0] += -0.07956751303949783;
                      }
                    } else {
                      result[0] += -0.09212811137057765;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.849175214767456943) ) ) {
                    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.08422076844394089;
                    } else {
                      result[0] += 0.005984819509362663;
                    }
                  } else {
                    result[0] += 0.05782530739458672;
                  }
                }
              } else {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                        result[0] += -0.023862669571034383;
                      } else {
                        result[0] += -0.12886698141265038;
                      }
                    } else {
                      result[0] += 0.06893612389375417;
                    }
                  } else {
                    result[0] += 0.03191854253661423;
                  }
                } else {
                  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)5.500000000000000888) ) ) {
                    result[0] += 0.009409749737600288;
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
                      result[0] += -0.037557279456931644;
                    } else {
                      result[0] += 0.3231016445315307;
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
                if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.094205617904663974) ) ) {
                    result[0] += 0.030523628209257882;
                  } else {
                    result[0] += -0.06623752233077655;
                  }
                } else {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                    result[0] += 0.0804420461988395;
                  } else {
                    result[0] += -0.07657538511762012;
                  }
                }
              } else {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                  result[0] += -0.015627316860390605;
                } else {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                    result[0] += 0.11254454400045993;
                  } else {
                    result[0] += -0.09446162267420184;
                  }
                }
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
        result[0] += 0.02365726648781677;
      } else {
        result[0] += 0.13521859944573072;
      }
    }
  } else {
    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.607751369476319248) ) ) {
        if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
          result[0] += -0.0062570156678395834;
        } else {
          result[0] += 0.10349747415883624;
        }
      } else {
        result[0] += -0.04810144731380653;
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.481347560882569248) ) ) {
        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.208071470260621005) ) ) {
          if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += 0.018968269531188377;
          } else {
            result[0] += -0.0493662774231259;
          }
        } else {
          result[0] += 0.08309069438861733;
        }
      } else {
        result[0] += -0.0012212596295428065;
      }
    }
  }
  if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)24.00000000000000355) ) ) {
    if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)4.500000000000000888) ) ) {
      if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
          if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.311204195022583896) ) ) {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += 0.008337363252290191;
                  } else {
                    if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.218042135238648349) ) ) {
                      if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                          result[0] += 0.28144677275665003;
                        } else {
                          result[0] += 0.10593893519026015;
                        }
                      } else {
                        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                          result[0] += 0.020842459482841174;
                        } else {
                          result[0] += 0.10062477623745124;
                        }
                      }
                    } else {
                      result[0] += 0.0009406910847599599;
                    }
                  }
                } else {
                  result[0] += -0.04666517075713164;
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.671854496002199042) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.978769779205324042) ) ) {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.00564580472062818;
                    } else {
                      result[0] += 0.05966851079804735;
                    }
                  } else {
                    if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( UNLIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += -0.030421406912519135;
                      } else {
                        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                          result[0] += -0.19279202034099774;
                        } else {
                          result[0] += -0.0964522076232659;
                        }
                      }
                    } else {
                      result[0] += 0.0027432900939777156;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.01392666360372277;
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)1.242453336715698464) ) ) {
                      result[0] += -0.047856327556802636;
                    } else {
                      result[0] += -8.134836825603398e-05;
                    }
                  }
                }
              }
            } else {
              result[0] += 0.025507839604557877;
            }
          } else {
            if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.671254158020020419) ) ) {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += 0.09805243013392252;
                } else {
                  result[0] += 0.014218529012894297;
                }
              } else {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.445705175399781162) ) ) {
                  result[0] += -0.0963202904464054;
                } else {
                  result[0] += -0.00507630617620047;
                }
              }
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.397998809814454013) ) ) {
                if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                      result[0] += -0.016307142333345515;
                    } else {
                      result[0] += 0.03746455020067492;
                    }
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.284418344497681552) ) ) {
                      result[0] += -0.07836509697346802;
                    } else {
                      result[0] += -0.013978457974820523;
                    }
                  }
                } else {
                  result[0] += 0.021939551086496215;
                }
              } else {
                if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                  result[0] += 0.0014677473243257131;
                } else {
                  result[0] += 0.06239287342423751;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
            if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)9.500000000000001776) ) ) {
              if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.009383988728450185;
                } else {
                  result[0] += -0.035246129749038846;
                }
              } else {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.357691764831543413) ) ) {
                    result[0] += -0.02519208395182725;
                  } else {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.484580039978028232) ) ) {
                        result[0] += 0.05932718527015301;
                      } else {
                        result[0] += 0.0046445578238453;
                      }
                    } else {
                      if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                        result[0] += 0.020347515315788284;
                      } else {
                        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                          if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            result[0] += -0.08222934392233577;
                          } else {
                            result[0] += 0.14714949400075047;
                          }
                        } else {
                          result[0] += 0.10209194002763163;
                        }
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                    if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.029068946838379794) ) ) {
                        result[0] += 0.09860608160925921;
                      } else {
                        if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += 0.03156980928140143;
                        } else {
                          result[0] += -0.10383740435892562;
                        }
                      }
                    } else {
                      result[0] += -0.1195354433476149;
                    }
                  } else {
                    result[0] += 0.0481042936641104;
                  }
                }
              }
            } else {
              result[0] += -0.06694042111679357;
            }
          } else {
            if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)8.500000000000001776) ) ) {
                result[0] += -0.02942564247381619;
              } else {
                if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
                  result[0] += 0.0633042489529003;
                } else {
                  result[0] += -0.02066291016584472;
                }
              }
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.827801465988160068) ) ) {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.284418344497681552) ) ) {
                    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += -0.00434027982261682;
                    } else {
                      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                        result[0] += 0.06662013139350347;
                      } else {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
                          result[0] += -0.04566663185417896;
                        } else {
                          result[0] += 0.05513448479898525;
                        }
                      }
                    }
                  } else {
                    result[0] += -0.0057370856630498965;
                  }
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.484580039978028232) ) ) {
                    result[0] += -0.062131536164238534;
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.909855604171753818) ) ) {
                      result[0] += 0.040487423079392015;
                    } else {
                      result[0] += -0.058572657812069474;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += -0.059878518955947806;
                } else {
                  result[0] += -0.0006563847677452245;
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.607751369476319248) ) ) {
            result[0] += -0.003242243452944443;
          } else {
            result[0] += -0.04795748869707797;
          }
        } else {
          if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.431901693344116655) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.924581527709961826) ) ) {
              result[0] += -0.06385601936400269;
            } else {
              result[0] += 0.08994564279232824;
            }
          } else {
            result[0] += -0.0012903154225332785;
          }
        }
      }
    } else {
      result[0] += 0.08151908608816172;
    }
  } else {
    result[0] += -0.09865698876261747;
  }
  if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
      if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
            if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += 0.007640835914982074;
            } else {
              if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.218042135238648349) ) ) {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.1432553086129116;
                } else {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                    result[0] += 0.017324174364756927;
                  } else {
                    result[0] += 0.08928804422483304;
                  }
                }
              } else {
                result[0] += 0.0015334150217584683;
              }
            }
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.05998366235601228;
            } else {
              result[0] += 0.006257878862481116;
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.671854496002199042) ) ) {
            if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
              result[0] += -0.07331429888142435;
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.978769779205324042) ) ) {
                result[0] += 0.0009260985174051261;
              } else {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.02923422744457146;
                  } else {
                    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                      result[0] += -0.18244238356147177;
                    } else {
                      result[0] += -0.08809643370816284;
                    }
                  }
                } else {
                  result[0] += 0.0035104614483127943;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += -0.010837428323999443;
            } else {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.516936540603638583) ) ) {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += -0.047558588116416695;
                  } else {
                    result[0] += 0.0007198688454648139;
                  }
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.484580039978028232) ) ) {
                    result[0] += -0.049754862068194794;
                  } else {
                    if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                      if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.284418344497681552) ) ) {
                          if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)4.500000000000000888) ) ) {
                            result[0] += -0.16631751605442366;
                          } else {
                            result[0] += -0.004626001601607424;
                          }
                        } else {
                          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                            result[0] += 0.044679362362584005;
                          } else {
                            result[0] += -0.0644777941698119;
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                          result[0] += 0.10793921254875719;
                        } else {
                          result[0] += -0.03436471533995194;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += -0.063798129078177;
                      } else {
                        result[0] += -0.006477613034823446;
                      }
                    }
                  }
                }
              } else {
                result[0] += 0.01786784276175901;
              }
            }
          }
        }
      } else {
        result[0] += 0.00734819016218333;
      }
    } else {
      if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)9.500000000000001776) ) ) {
          if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.938867926597595659) ) ) {
                if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
                  result[0] += 0.1113315591186232;
                } else {
                  result[0] += 0.023737013232216764;
                }
              } else {
                if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += 0.008456937062523344;
                } else {
                  result[0] += -0.04122613113108778;
                }
              }
            } else {
              result[0] += -0.035602784151173315;
            }
          } else {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
                result[0] += 0.09170960226200452;
              } else {
                result[0] += -0.07893388704962558;
              }
            } else {
              if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.849175214767456943) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.843275547027588779) ) ) {
                    if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += 0.054042288607371564;
                    } else {
                      result[0] += 0.004241155680437026;
                    }
                  } else {
                    result[0] += 0.0879197621325739;
                  }
                } else {
                  result[0] += -0.05600962019940553;
                }
              } else {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                  if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.88435244560241788) ) ) {
                      result[0] += -0.039729045640382654;
                    } else {
                      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                        result[0] += 0.0031954779122840755;
                      } else {
                        result[0] += 0.1122655892456745;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += 0.01270174786525216;
                      } else {
                        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.802696108818054643) ) ) {
                          result[0] += -0.03777994038928664;
                        } else {
                          result[0] += 0.10299255768342325;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                        result[0] += 0.058891750790074394;
                      } else {
                        result[0] += -0.07026927777790826;
                      }
                    }
                  }
                } else {
                  result[0] += -0.07794393342326245;
                }
              }
            }
          }
        } else {
          result[0] += -0.06356842940375242;
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.744781017303467685) ) ) {
          if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.154959201812744585) ) ) {
              result[0] += 0.11230389458753386;
            } else {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += -0.042370508628900044;
              } else {
                result[0] += 0.07505550412115727;
              }
            }
          } else {
            result[0] += 0.0027510536325702356;
          }
        } else {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
            result[0] += -0.04265197687771688;
          } else {
            if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.993164777755738193) ) ) {
                result[0] += -0.08861018467692498;
              } else {
                result[0] += -0.012288512567989131;
              }
            } else {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                  result[0] += -0.028205596530289558;
                } else {
                  result[0] += 0.04720200054781679;
                }
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.778982400894165927) ) ) {
                  result[0] += -0.020378041268323793;
                } else {
                  result[0] += 0.021188971634491238;
                }
              }
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.718933820724488193) ) ) {
        if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
          result[0] += -0.008653666029318673;
        } else {
          result[0] += 0.08901627066662242;
        }
      } else {
        result[0] += -0.057354162341528694;
      }
    } else {
      result[0] += -0.0016584827493588987;
    }
  }
  if ( LIKELY(  (data[42].missing != -1) && (data[42].fvalue <= (double)-1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
      if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.311204195022583896) ) ) {
          if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.481121778488159624) ) ) {
              if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
                if ( UNLIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.03964262153370826;
                } else {
                  result[0] += -0.17289946755424845;
                }
              } else {
                result[0] += 0.04601829936362417;
              }
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.511434078216553178) ) ) {
                result[0] += 0.09289978811033907;
              } else {
                result[0] += -0.0032673506746703734;
              }
            }
          } else {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.868834793567657693) ) ) {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.284418344497681552) ) ) {
                    result[0] += 0.003953516399920211;
                  } else {
                    if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.406644821166993076) ) ) {
                      result[0] += 0.12862136330202556;
                    } else {
                      result[0] += 0.023733568095637345;
                    }
                  }
                } else {
                  result[0] += 0.10540365481129917;
                }
              } else {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.044173099565575594;
                } else {
                  result[0] += 0.009125164947733775;
                }
              }
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.602003335952759233) ) ) {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                      if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += -0.031233885236857623;
                      } else {
                        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                          result[0] += -0.1585270707241332;
                        } else {
                          result[0] += -0.04777580423421654;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                        result[0] += 0.017155304306641625;
                      } else {
                        result[0] += -0.05272032248791933;
                      }
                    }
                  } else {
                    result[0] += -0.17091916687363148;
                  }
                } else {
                  result[0] += 0.002701543410525215;
                }
              } else {
                if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                    if ( LIKELY(  (data[43].missing != -1) && (data[43].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                      result[0] += -0.00020074927293922394;
                    } else {
                      if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.651049375534058505) ) ) {
                          result[0] += -0.1187257159323262;
                        } else {
                          if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            result[0] += -0.0720423544190034;
                          } else {
                            result[0] += 0.06318654363090365;
                          }
                        }
                      } else {
                        result[0] += 0.006496041905439617;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.05909916589576791;
                    } else {
                      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.433569431304932529) ) ) {
                        if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                          result[0] += 0.09331911038989908;
                        } else {
                          result[0] += -0.08369857744268908;
                        }
                      } else {
                        result[0] += -0.08676757004409445;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.694163918495178667) ) ) {
                        result[0] += 0.003217357167567705;
                      } else {
                        result[0] += -0.031800644009838855;
                      }
                    } else {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.920601367950440341) ) ) {
                        result[0] += 0.06007199184311236;
                      } else {
                        result[0] += -0.15856001022939145;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)4.500000000000000888) ) ) {
                      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                        result[0] += 0.019901910582230356;
                      } else {
                        result[0] += -0.04399311434161398;
                      }
                    } else {
                      result[0] += -0.07630973297574845;
                    }
                  }
                }
              }
            }
          }
        } else {
          result[0] += 0.024617812705160173;
        }
      } else {
        result[0] += 0.006381815110857352;
      }
    } else {
      if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.715336322784424716) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.208071470260621005) ) ) {
              result[0] += 0.008026845130923452;
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.255632162094117099) ) ) {
                if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)4.500000000000000888) ) ) {
                  result[0] += 0.03956704362044228;
                } else {
                  result[0] += 0.13019800729683353;
                }
              } else {
                result[0] += 0.001544495735759194;
              }
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.029068946838379794) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.88435244560241788) ) ) {
                result[0] += -0.10797539349210517;
              } else {
                result[0] += -0.02279705774509354;
              }
            } else {
              result[0] += 0.003364821122690373;
            }
          }
        } else {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.357691764831543413) ) ) {
            result[0] += -0.04864310034231089;
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                  result[0] += 0.0474298477463855;
                } else {
                  result[0] += -0.045624895066847665;
                }
              } else {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.516936540603638583) ) ) {
                    result[0] += -0.031141262420045796;
                  } else {
                    result[0] += 0.039064919971316815;
                  }
                } else {
                  result[0] += 0.0853198120653867;
                }
              }
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += 0.03454080963846261;
                } else {
                  result[0] += -0.11505201789166522;
                }
              } else {
                result[0] += 0.04914072473329611;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += -0.018511645009072694;
        } else {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.827801465988160068) ) ) {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.284418344497681552) ) ) {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += -0.003218856605901204;
                } else {
                  result[0] += 0.05357938877398464;
                }
              } else {
                result[0] += -0.005791333938077924;
              }
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.484580039978028232) ) ) {
                result[0] += -0.06255453705638313;
              } else {
                result[0] += 0.017659207907352398;
              }
            }
          } else {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += -0.03540137207278339;
              } else {
                result[0] += 0.006068723987852045;
              }
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.255632162094117099) ) ) {
                result[0] += 0.005236571832289379;
              } else {
                result[0] += 0.08036851524700757;
              }
            }
          }
        }
      }
    }
  } else {
    result[0] += -0.0020611330226903697;
  }
  if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
    result[0] += 0.002708178599323018;
  } else {
    if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.795762062072754794) ) ) {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.285887241363526279) ) ) {
            result[0] += -0.007321557956521996;
          } else {
            result[0] += -0.06957018715770952;
          }
        } else {
          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.516936540603638583) ) ) {
                result[0] += -0.022375854144784005;
              } else {
                result[0] += -0.08808341051395793;
              }
            } else {
              result[0] += 0.01936536594684251;
            }
          } else {
            result[0] += 0.01791083399943362;
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.95906782150268732) ) ) {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
            if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.0003439901127975499;
            } else {
              result[0] += -0.08144324609054761;
            }
          } else {
            result[0] += 0.04684314841622053;
          }
        } else {
          result[0] += -0.039704204394260194;
        }
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.142630577087403232) ) ) {
        if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
              if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += 0.08602167226426347;
                  } else {
                    result[0] += -0.024912576726723153;
                  }
                } else {
                  result[0] += 0.07052265241274319;
                }
              } else {
                result[0] += 0.005798509821087086;
              }
            } else {
              result[0] += 0.013025832750570544;
            }
          } else {
            if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
              result[0] += -0.07083435322545067;
            } else {
              result[0] += 0.020426591993779344;
            }
          }
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.694163918495178667) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.790835380554201) ) ) {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)46.00000000000000711) ) ) {
                result[0] += 0.0777983903454979;
              } else {
                result[0] += 0.007862976325797823;
              }
            } else {
              if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)6.000000000000000888) ) ) {
                if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.589026927947998269) ) ) {
                  result[0] += -0.006497743842839368;
                } else {
                  result[0] += -0.08177901754504063;
                }
              } else {
                result[0] += -0.037799204999173316;
              }
            }
          } else {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += 0.046942904277459216;
            } else {
              result[0] += -0.006889117111555112;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)40.00000000000000711) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.124530076980591708) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.778982400894165927) ) ) {
              result[0] += -0.001161946783425108;
            } else {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += -0.019292995693967988;
              } else {
                result[0] += -0.0786454408318243;
              }
            }
          } else {
            result[0] += -0.07337354154893846;
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.694163918495178667) ) ) {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
              result[0] += 0.05968055770647897;
            } else {
              result[0] += -0.026551528016250527;
            }
          } else {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.651049375534058505) ) ) {
                result[0] += 0.01439663558674284;
              } else {
                result[0] += -0.03313190242565386;
              }
            } else {
              if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.142630577087403232) ) ) {
                  result[0] += -0.08807778810502913;
                } else {
                  result[0] += -0.0025237596093304303;
                }
              } else {
                if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += -0.05354121602745241;
                } else {
                  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)6.000000000000000888) ) ) {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.284074544906617099) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.481347560882569248) ) ) {
                        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                          result[0] += -0.09031729874033019;
                        } else {
                          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.208071470260621005) ) ) {
                            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                              result[0] += 0.06489794593139142;
                            } else {
                              result[0] += -0.05626663582276878;
                            }
                          } else {
                            result[0] += 0.10284755263992679;
                          }
                        }
                      } else {
                        if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
                          result[0] += 0.016902505930349466;
                        } else {
                          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
                            result[0] += -0.06502462515037898;
                          } else {
                            result[0] += 0.02386120843409604;
                          }
                        }
                      }
                    } else {
                      result[0] += -0.009577509675805482;
                    }
                  } else {
                    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                        if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
                          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.909855604171753818) ) ) {
                            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.172047138214112216) ) ) {
                              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                                result[0] += 0.06010697949525358;
                              } else {
                                if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                                  result[0] += -0.017782988250163324;
                                } else {
                                  result[0] += 0.0874272492094795;
                                }
                              }
                            } else {
                              result[0] += 0.0714834947227307;
                            }
                          } else {
                            result[0] += 0.000867911551186029;
                          }
                        } else {
                          result[0] += -0.05170486772169585;
                        }
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.284418344497681552) ) ) {
                          result[0] += -0.09518951669576524;
                        } else {
                          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)46.00000000000000711) ) ) {
                            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.673553824424744096) ) ) {
                              result[0] += -0.07325920816175628;
                            } else {
                              result[0] += 0.039312856226608725;
                            }
                          } else {
                            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.623839378356934482) ) ) {
                              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.718933820724488193) ) ) {
                                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
                                  result[0] += 0.00800628164268097;
                                } else {
                                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.433569431304932529) ) ) {
                                    if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                                      result[0] += -0.15769277374789137;
                                    } else {
                                      result[0] += -0.03506618235905626;
                                    }
                                  } else {
                                    result[0] += 0.006400106073339607;
                                  }
                                }
                              } else {
                                if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                                  result[0] += -0.021617966112466924;
                                } else {
                                  result[0] += 0.08096103710749686;
                                }
                              }
                            } else {
                              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.718933820724488193) ) ) {
                                result[0] += -0.17885035993032428;
                              } else {
                                result[0] += -0.027967579383285382;
                              }
                            }
                          }
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
                        result[0] += 0.0034747145891060164;
                      } else {
                        result[0] += -0.029130173231289566;
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
  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
    if ( UNLIKELY(  (data[43].missing != -1) && (data[43].fvalue <= (double)-1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.48918962478637873) ) ) {
          if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2252.000000000000455) ) ) {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.051912069320679599) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                result[0] += -0.004630906816007855;
              } else {
                result[0] += 0.013276831133842898;
              }
            } else {
              if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.009826072332728456;
              } else {
                result[0] += -0.02946760371462519;
              }
            }
          } else {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.77496147155761896) ) ) {
                result[0] += 0.0024378156902561342;
              } else {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.911822915077209917) ) ) {
                  result[0] += 0.060485858858413837;
                } else {
                  result[0] += 0.01813824446930794;
                }
              }
            } else {
              result[0] += -0.0002985277464284072;
            }
          }
        } else {
          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
            result[0] += 0.013073781171629956;
          } else {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += 0.07614510826530692;
            } else {
              result[0] += -0.06013759575504464;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.736135363578796831) ) ) {
          result[0] += 0.05472627863004609;
        } else {
          if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.284418344497681552) ) ) {
              result[0] += 0.003647389629710302;
            } else {
              result[0] += -0.0671958701355029;
            }
          } else {
            result[0] += 0.014325304805372325;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
          if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
              if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                result[0] += -0.10160259786281273;
              } else {
                result[0] += 0.01663739174878801;
              }
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.868834793567657693) ) ) {
                result[0] += -0.061169121514481666;
              } else {
                result[0] += 0.06634549733824356;
              }
            }
          } else {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.484580039978028232) ) ) {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.313157320022583896) ) ) {
                  if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)46.00000000000000711) ) ) {
                      result[0] += 0.026769036663362108;
                    } else {
                      result[0] += -0.036108831364476186;
                    }
                  } else {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.827801465988160068) ) ) {
                      result[0] += 0.013845989961500525;
                    } else {
                      result[0] += -0.09721182651824872;
                    }
                  }
                } else {
                  result[0] += 0.020723080178663623;
                }
              } else {
                if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                  result[0] += -0.015838765540393297;
                } else {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                    result[0] += -0.03977393468324342;
                  } else {
                    result[0] += 0.0074378813570239;
                  }
                }
              }
            } else {
              result[0] += -0.06738908708989999;
            }
          }
        } else {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.56849193572998225) ) ) {
              result[0] += -0.047385038461712534;
            } else {
              result[0] += 0.01745859438004819;
            }
          } else {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
              result[0] += -0.0260960000921197;
            } else {
              result[0] += 0.07025142180865046;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.142630577087403232) ) ) {
          if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)2.238668441772461382) ) ) {
                result[0] += -0.0033761791426322207;
              } else {
                result[0] += 0.0789906372164394;
              }
            } else {
              result[0] += -0.06888993481367621;
            }
          } else {
            result[0] += -0.03521219911728463;
          }
        } else {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.795762062072754794) ) ) {
                result[0] += -0.009541939274306124;
              } else {
                result[0] += -0.05133433269560703;
              }
            } else {
              result[0] += 0.01565244788088579;
            }
          } else {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.843275547027588779) ) ) {
              if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.031032371130760506;
                    } else {
                      result[0] += 0.07972166389916352;
                    }
                  } else {
                    result[0] += -0.0214055675746933;
                  }
                } else {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.718933820724488193) ) ) {
                      result[0] += -0.012811615208355442;
                    } else {
                      result[0] += -0.09268037397443463;
                    }
                  } else {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
                      result[0] += 0.0026551422350273466;
                    } else {
                      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.339395284652710849) ) ) {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                          result[0] += -0.0021386062612888546;
                        } else {
                          result[0] += 0.05483277492160158;
                        }
                      } else {
                        result[0] += -0.05689793840743376;
                      }
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.019094640721096564;
                } else {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.802901029586792436) ) ) {
                    if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.04615460098193711;
                    } else {
                      result[0] += 0.07686314394318909;
                    }
                  } else {
                    if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                        result[0] += -0.06782415165555054;
                      } else {
                        result[0] += 0.004212152394163979;
                      }
                    } else {
                      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
                          result[0] += -0.04194521347070853;
                        } else {
                          result[0] += 0.10128953466417955;
                        }
                      } else {
                        result[0] += -0.05171366120497392;
                      }
                    }
                  }
                }
              }
            } else {
              result[0] += 0.01611456176763409;
            }
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
      result[0] += -0.0016850142785480408;
    } else {
      if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)9.500000000000001776) ) ) {
        if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
          result[0] += -0.078593075112503;
        } else {
          result[0] += -0.019581564040874952;
        }
      } else {
        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)10.50000000000000178) ) ) {
          if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += 0.026651346054381492;
          } else {
            result[0] += 0.1329353728105974;
          }
        } else {
          result[0] += -0.041343531013286396;
        }
      }
    }
  }
  if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
    result[0] += 0.002430842691862018;
  } else {
    if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.795762062072754794) ) ) {
          result[0] += -0.007402248736265328;
        } else {
          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += -0.04239468251499712;
            } else {
              result[0] += 0.017230724059223573;
            }
          } else {
            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
              result[0] += 0.020378520139573583;
            } else {
              result[0] += -0.00736965801502154;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.95906782150268732) ) ) {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += 0.001132217175332595;
            } else {
              result[0] += -0.077114061674307;
            }
          } else {
            result[0] += 0.040019870160066695;
          }
        } else {
          result[0] += -0.03709642274028794;
        }
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.142630577087403232) ) ) {
        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.94957673549652144) ) ) {
            if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
                  result[0] += 0.04906246782310554;
                } else {
                  result[0] += 0.013101692922028951;
                }
              } else {
                if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                  result[0] += -0.04192643722282641;
                } else {
                  result[0] += 0.05777715156283919;
                }
              }
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.694163918495178667) ) ) {
                if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                  result[0] += 0.005377315997943169;
                } else {
                  result[0] += -0.03864956167705644;
                }
              } else {
                if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                  result[0] += 0.027229981331450978;
                } else {
                  if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                    result[0] += 0.14084849969463603;
                  } else {
                    result[0] += 0.044021061959786445;
                  }
                }
              }
            }
          } else {
            result[0] += 0.13234656282365384;
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.154959201812744585) ) ) {
            result[0] += -0.06302296834564063;
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.95659637451172053) ) ) {
              result[0] += 0.017965842299994986;
            } else {
              if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += 0.0008327474712067714;
              } else {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.04680543403504273;
                } else {
                  result[0] += 0.008671571786317207;
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)40.00000000000000711) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.909855604171753818) ) ) {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.651049375534058505) ) ) {
              result[0] += 0.0024353821433814333;
            } else {
              result[0] += -0.02439588505650164;
            }
          } else {
            result[0] += -0.06337305106638845;
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.694163918495178667) ) ) {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
              result[0] += 0.05398938674995916;
            } else {
              result[0] += -0.026167343259485505;
            }
          } else {
            if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.607751369476319248) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.357691764831543413) ) ) {
                  result[0] += 0.10725352971864768;
                } else {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
                    result[0] += 0.018592728308022465;
                  } else {
                    result[0] += -0.01336011521344354;
                  }
                }
              } else {
                result[0] += -0.037052165179019284;
              }
            } else {
              if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.276817798614503729) ) ) {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                    result[0] += -0.06480978521578724;
                  } else {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                      result[0] += 0.07174710716837308;
                    } else {
                      if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += -0.015322679831828806;
                      } else {
                        result[0] += -0.13437852992031152;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                    if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                      result[0] += -0.06437179678843975;
                    } else {
                      result[0] += 0.07042900968136398;
                    }
                  } else {
                    if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += 0.014236010249816557;
                    } else {
                      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.778982400894165927) ) ) {
                            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.397998809814454013) ) ) {
                              result[0] += 0.02965055293440534;
                            } else {
                              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.909855604171753818) ) ) {
                                result[0] += -0.02987902240943502;
                              } else {
                                result[0] += -0.14175283872249209;
                              }
                            }
                          } else {
                            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.843275547027588779) ) ) {
                              result[0] += 0.06686246401927895;
                            } else {
                              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.827801465988160068) ) ) {
                                result[0] += -0.09184751189563667;
                              } else {
                                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.516936540603638583) ) ) {
                                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.172047138214112216) ) ) {
                                    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                                      result[0] += 0.07539441407475693;
                                    } else {
                                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.743881702423096591) ) ) {
                                        result[0] += -0.08018670042006097;
                                      } else {
                                        result[0] += 0.031335284981247144;
                                      }
                                    }
                                  } else {
                                    result[0] += 0.08018479885997792;
                                  }
                                } else {
                                  result[0] += -0.023619211515484184;
                                }
                              }
                            }
                          }
                        } else {
                          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.433569431304932529) ) ) {
                            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
                              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.284418344497681552) ) ) {
                                result[0] += -0.08294072434791804;
                              } else {
                                result[0] += 0.02915531471002547;
                              }
                            } else {
                              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                                result[0] += -0.13188856403403273;
                              } else {
                                result[0] += -0.041341706932831526;
                              }
                            }
                          } else {
                            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)46.00000000000000711) ) ) {
                              result[0] += 0.0479235771281734;
                            } else {
                              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.623839378356934482) ) ) {
                                result[0] += 0.009271433935812406;
                              } else {
                                result[0] += -0.06647368015581338;
                              }
                            }
                          }
                        }
                      } else {
                        result[0] += -0.017025458677198405;
                      }
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
                  result[0] += -0.00613071798774468;
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.993164777755738193) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
                      result[0] += 0.014131606074440809;
                    } else {
                      result[0] += -0.09273180927423241;
                    }
                  } else {
                    result[0] += 0.05888335473965324;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
    result[0] += 0.0037619681466040944;
  } else {
    if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
      if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
        result[0] += -0.0471967986501358;
      } else {
        if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.285887241363526279) ) ) {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += -0.009971978193565404;
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.94957673549652144) ) ) {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                  result[0] += 0.03220807795530463;
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.208071470260621005) ) ) {
                    result[0] += -0.0339431093441596;
                  } else {
                    result[0] += 0.043068983038105586;
                  }
                }
              } else {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
                  if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                    if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.05821206532194931;
                    } else {
                      if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                        result[0] += 0.010325063689031523;
                      } else {
                        result[0] += -0.03491561892609959;
                      }
                    }
                  } else {
                    result[0] += -0.0015746270420982846;
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
                    if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
                      if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2415.000000000000455) ) ) {
                        result[0] += 0.00764779572122792;
                      } else {
                        result[0] += 0.09761036931488008;
                      }
                    } else {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.778982400894165927) ) ) {
                        result[0] += 0.004035465135374495;
                      } else {
                        if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                          result[0] += -0.11361324043450059;
                        } else {
                          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.192109584808350498) ) ) {
                            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.208071470260621005) ) ) {
                                result[0] += -0.00840575472915016;
                              } else {
                                result[0] += -0.13088937776491227;
                              }
                            } else {
                              result[0] += 0.004665307838980928;
                            }
                          } else {
                            result[0] += 0.10263781573478543;
                          }
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.397998809814454013) ) ) {
                      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                        if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2415.000000000000455) ) ) {
                          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
                            result[0] += 0.033467854938933735;
                          } else {
                            result[0] += 0.12181045476624269;
                          }
                        } else {
                          result[0] += 0.11242294225722405;
                        }
                      } else {
                        if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                          result[0] += 0.05274048246389776;
                        } else {
                          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                            result[0] += -0.13356039209051343;
                          } else {
                            result[0] += -0.03105937070380645;
                          }
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.95211219787597834) ) ) {
                          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                            result[0] += 0.07088025520019627;
                          } else {
                            result[0] += -0.07701036201869535;
                          }
                        } else {
                          if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
                            result[0] += 0.05139491035730753;
                          } else {
                            result[0] += -0.025977835036452443;
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += 0.02545370179181581;
                        } else {
                          result[0] += -0.016907915388418723;
                        }
                      }
                    }
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)46.00000000000000711) ) ) {
                result[0] += -0.1012239510803733;
              } else {
                result[0] += -0.03371197220756041;
              }
            } else {
              result[0] += 0.034612651926277944;
            }
          }
        } else {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.778982400894165927) ) ) {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.718933820724488193) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.284418344497681552) ) ) {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                    if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                      if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)6.023992538452149326) ) ) {
                        result[0] += 0.027908374398663782;
                      } else {
                        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                          result[0] += 0.0920072763497437;
                        } else {
                          result[0] += -0.021205064706685362;
                        }
                      }
                    } else {
                      result[0] += -0.0371374135352302;
                    }
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.154959201812744585) ) ) {
                      result[0] += -0.08298651909744582;
                    } else {
                      result[0] += 0.012772997718148483;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.029068946838379794) ) ) {
                    if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.037688311909612915;
                    } else {
                      result[0] += -0.04834383214725371;
                    }
                  } else {
                    result[0] += 0.036658869334610315;
                  }
                }
              } else {
                result[0] += 0.10193433227780245;
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.484580039978028232) ) ) {
                result[0] += -0.0736567051926663;
              } else {
                result[0] += 0.023414433556583014;
              }
            }
          } else {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.172047138214112216) ) ) {
                result[0] += -0.0227826828456971;
              } else {
                result[0] += -0.07772913008459259;
              }
            } else {
              result[0] += 0.0027022822296221983;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.938867926597595659) ) ) {
          result[0] += 0.023060795538527662;
        } else {
          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.297559976577759233) ) ) {
              result[0] += 0.028492177571037454;
            } else {
              result[0] += -0.04696295051879107;
            }
          } else {
            result[0] += 0.020513519442037373;
          }
        }
      } else {
        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)9.500000000000001776) ) ) {
          if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)4.500000000000000888) ) ) {
            result[0] += 0.003632147114303706;
          } else {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.909855604171753818) ) ) {
                result[0] += 0.0077437805279098135;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.467917680740357333) ) ) {
                  result[0] += 0.005230510752028605;
                } else {
                  result[0] += -0.055202660291681166;
                }
              }
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.909855604171753818) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.938988685607911933) ) ) {
                  result[0] += 0.00956989639303251;
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.284418344497681552) ) ) {
                    result[0] += -0.043372059996350965;
                  } else {
                    result[0] += -0.013244485019026989;
                  }
                }
              } else {
                result[0] += 0.011192867104952849;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)10.50000000000000178) ) ) {
            result[0] += 0.041768603234540354;
          } else {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
              result[0] += 0.10715531035135897;
            } else {
              result[0] += -0.03660967265616075;
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)10.00000000000000178) ) ) {
    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.339395284652710849) ) ) {
      if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.87008237838745206) ) ) {
          result[0] += 0.0011179868772737034;
        } else {
          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
            result[0] += 0.0071097019794463066;
          } else {
            result[0] += 0.038649498962540337;
          }
        }
      } else {
        result[0] += -0.038140156492357524;
      }
    } else {
      if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.172047138214112216) ) ) {
          if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)257681260544.0000305) ) ) {
            result[0] += -0.011372355919121486;
          } else {
            result[0] += 0.10384252158481887;
          }
        } else {
          result[0] += 0.013752312411327669;
        }
      } else {
        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
          result[0] += 0.02815909014736418;
        } else {
          result[0] += 0.0721269122881995;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
      if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
        result[0] += -0.04763701885212187;
      } else {
        result[0] += -0.0033834357147827817;
      }
    } else {
      if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)9.500000000000001776) ) ) {
        if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.135017871856690341) ) ) {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += 0.00623011184461341;
              } else {
                result[0] += 0.05920089939079501;
              }
            } else {
              if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.002260344612032035;
                } else {
                  if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
                    result[0] += -0.04017926335644207;
                  } else {
                    result[0] += 0.08167101686672681;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.700598716735840066) ) ) {
                  result[0] += 0.009836449925154617;
                } else {
                  result[0] += -0.028436071393594865;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.770631790161133257) ) ) {
              result[0] += -0.08228592998120077;
            } else {
              result[0] += 0.0009581357840405475;
            }
          }
        } else {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.357691764831543413) ) ) {
            result[0] += 0.026573449232673543;
          } else {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.142630577087403232) ) ) {
                if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += 0.12333370130065027;
                } else {
                  result[0] += -0.01246313417349219;
                }
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.261864185333252841) ) ) {
                  if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                    if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                      result[0] += 0.19901095430625157;
                    } else {
                      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.849175214767456943) ) ) {
                        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.48738741874694913) ) ) {
                              result[0] += 0.003525620726726129;
                            } else {
                              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.534971714019776279) ) ) {
                                result[0] += -0.26706833701116045;
                              } else {
                                result[0] += -0.03972465394672087;
                              }
                            }
                          } else {
                            result[0] += 0.04887970171466083;
                          }
                        } else {
                          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.255632162094117099) ) ) {
                            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.208071470260621005) ) ) {
                              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.029068946838379794) ) ) {
                                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.088880300521851474) ) ) {
                                  result[0] += -0.09891418544423956;
                                } else {
                                  result[0] += 0.02612510702950657;
                                }
                              } else {
                                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                                  result[0] += -0.09836005494641714;
                                } else {
                                  result[0] += 0.03111523421201065;
                                }
                              }
                            } else {
                              result[0] += -0.12934535642832534;
                            }
                          } else {
                            result[0] += -0.007799842038899623;
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                          result[0] += -0.07891839165350528;
                        } else {
                          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.142630577087403232) ) ) {
                            result[0] += -0.19566977579853664;
                          } else {
                            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                              result[0] += 0.04275888033319661;
                            } else {
                              result[0] += -0.0752919472343253;
                            }
                          }
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.284418344497681552) ) ) {
                        result[0] += 0.07899809955133147;
                      } else {
                        result[0] += -0.022549898347402297;
                      }
                    } else {
                      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.993164777755738193) ) ) {
                          if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            result[0] += -0.07339873389074784;
                          } else {
                            result[0] += 0.020077152891983974;
                          }
                        } else {
                          result[0] += -0.06714329326522532;
                        }
                      } else {
                        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.484580039978028232) ) ) {
                          if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
                            result[0] += 0.022875152696081834;
                          } else {
                            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.172047138214112216) ) ) {
                                result[0] += 0.11410357853501046;
                              } else {
                                result[0] += -0.0245602101723829;
                              }
                            } else {
                              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)46.00000000000000711) ) ) {
                                  result[0] += 0.05444937905808888;
                                } else {
                                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                                    result[0] += -0.20364004115020476;
                                  } else {
                                    result[0] += -0.06227976278024766;
                                  }
                                }
                              } else {
                                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.172047138214112216) ) ) {
                                  result[0] += -0.10585707581317826;
                                } else {
                                  result[0] += 0.025429652519354397;
                                }
                              }
                            }
                          }
                        } else {
                          result[0] += 0.018523691663690327;
                        }
                      }
                    }
                  }
                } else {
                  result[0] += -0.009932882799043023;
                }
              }
            } else {
              if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)15.02900028228759943) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.172047138214112216) ) ) {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                      result[0] += 0.09255794141665688;
                    } else {
                      result[0] += 0.022672154617748724;
                    }
                  } else {
                    result[0] += -0.02274153704877243;
                  }
                } else {
                  result[0] += -0.25786821400505494;
                }
              } else {
                result[0] += 0.006436639885744765;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)10.50000000000000178) ) ) {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.861792564392090288) ) ) {
            result[0] += -0.019630311717195138;
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.607751369476319248) ) ) {
              result[0] += 0.022360009092782254;
            } else {
              result[0] += 0.0855628070182206;
            }
          }
        } else {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
            result[0] += 0.09642887124368715;
          } else {
            result[0] += -0.0322043876739747;
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)10.00000000000000178) ) ) {
    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.339395284652710849) ) ) {
      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.699081301689148393) ) ) {
        if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.793600082397461826) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)0.8958797454833985485) ) ) {
            result[0] += 0.0014147859194833736;
          } else {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.700598716735840066) ) ) {
              result[0] += 0.11874316593917428;
            } else {
              result[0] += 0.04744044975797729;
            }
          }
        } else {
          result[0] += -0.007074256898265796;
        }
      } else {
        if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
          result[0] += 0.003475986264945006;
        } else {
          result[0] += -0.025089538115535312;
        }
      }
    } else {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
        result[0] += 0.00814332787678274;
      } else {
        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
          result[0] += 0.021561302421823563;
        } else {
          result[0] += 0.06707898339003379;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
      if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
        result[0] += -0.04808624249181671;
      } else {
        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2252.000000000000455) ) ) {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.938867926597595659) ) ) {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += 0.06603545903295979;
            } else {
              result[0] += -0.002019578401280806;
            }
          } else {
            result[0] += -0.0193992457977218;
          }
        } else {
          result[0] += -0.001994367372498121;
        }
      }
    } else {
      if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)9.500000000000001776) ) ) {
          if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)1.354025125503540261) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.055836200714113104) ) ) {
                if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.909855604171753818) ) ) {
                      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.607751369476319248) ) ) {
                        result[0] += -0.03564887225453089;
                      } else {
                        result[0] += 0.020510109575546456;
                      }
                    } else {
                      result[0] += -0.09236470564014021;
                    }
                  } else {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.868834793567657693) ) ) {
                      result[0] += 0.02120467047811471;
                    } else {
                      result[0] += 0.09724342583758128;
                    }
                  }
                } else {
                  result[0] += -0.06104892831988892;
                }
              } else {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.48918962478637873) ) ) {
                      result[0] += -0.014721104765202801;
                    } else {
                      result[0] += -0.046163252700086754;
                    }
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.357691764831543413) ) ) {
                      result[0] += 0.11840548368963132;
                    } else {
                      result[0] += 0.027124353195954177;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
                      result[0] += 0.005229411879059274;
                    } else {
                      result[0] += -0.05379131996370162;
                    }
                  } else {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                      if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += -0.027182994497215413;
                      } else {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.208071470260621005) ) ) {
                          result[0] += 0.0035576061210413736;
                        } else {
                          result[0] += -0.025541363273058462;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)4.500000000000000888) ) ) {
                        result[0] += 0.06976796316089527;
                      } else {
                        result[0] += 0.0011268748248463424;
                      }
                    }
                  }
                }
              }
            } else {
              result[0] += -0.15993636085971036;
            }
          } else {
            if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.589026927947998269) ) ) {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.201062679290773261) ) ) {
                  result[0] += -0.05060324552844582;
                } else {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.302512168884278232) ) ) {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                      result[0] += 0.0058906614779867085;
                    } else {
                      if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.613531112670900214) ) ) {
                          result[0] += 0.019197137454116914;
                        } else {
                          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.384830474853516513) ) ) {
                            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.088880300521851474) ) ) {
                              result[0] += -0.024733003287555694;
                            } else {
                              result[0] += -0.08860183819219718;
                            }
                          } else {
                            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.029068946838379794) ) ) {
                              result[0] += 0.0302927521155431;
                            } else {
                              result[0] += -0.03031138163327726;
                            }
                          }
                        }
                      } else {
                        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                          if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            result[0] += 0.07957001146354582;
                          } else {
                            result[0] += 0.008955336479747467;
                          }
                        } else {
                          result[0] += -0.09110102825379475;
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                      result[0] += -0.05841505288077758;
                    } else {
                      if ( UNLIKELY(  (data[43].missing != -1) && (data[43].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.449861526489258257) ) ) {
                          result[0] += -0.005941115410779046;
                        } else {
                          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.918272972106934482) ) ) {
                            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.993164777755738193) ) ) {
                              result[0] += 0.03663182652720657;
                            } else {
                              if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                                result[0] += 0.023364173138991275;
                              } else {
                                result[0] += -0.11371943697162944;
                              }
                            }
                          } else {
                            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.924581527709961826) ) ) {
                              result[0] += 0.06862245938522799;
                            } else {
                              result[0] += 0.010539110885351739;
                            }
                          }
                        }
                      } else {
                        result[0] += -0.0007131959873252914;
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                  result[0] += -0.0840675125014256;
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.827801465988160068) ) ) {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)6.117118597030640537) ) ) {
                      result[0] += 0.005686050058384582;
                    } else {
                      result[0] += -0.12027143969606678;
                    }
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.516936540603638583) ) ) {
                      result[0] += -0.08372497099058544;
                    } else {
                      if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                        if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                            result[0] += -0.03737474261220185;
                          } else {
                            result[0] += -0.16623373396388402;
                          }
                        } else {
                          result[0] += 0.00742578251093482;
                        }
                      } else {
                        result[0] += 0.0022301785738837164;
                      }
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.142630577087403232) ) ) {
                result[0] += -0.029998665884712103;
              } else {
                result[0] += 0.03559342268258069;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.918272972106934482) ) ) {
            result[0] += -0.0026103750681542213;
          } else {
            result[0] += 0.04360384866239934;
          }
        }
      } else {
        result[0] += 0.00496179284161205;
      }
    }
  }
  if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)3.000000000000000444) ) ) {
    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)1.994492053985595925) ) ) {
      if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.835998296737671787) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)0.8958797454833985485) ) ) {
          result[0] += 0.0014067503016172262;
        } else {
          result[0] += 0.07717825334989847;
        }
      } else {
        result[0] += -0.0036926495967039086;
      }
    } else {
      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.851041555404663974) ) ) {
          result[0] += 0.0033663635099663285;
        } else {
          result[0] += -0.032821979820836567;
        }
      } else {
        if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)46.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)8.500000000000001776) ) ) {
              result[0] += 0.0855293266801469;
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.605039834976196733) ) ) {
                result[0] += 0.20357868565186352;
              } else {
                result[0] += -0.059584160222288254;
              }
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.607751369476319248) ) ) {
              if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.158952236175537998) ) ) {
                  result[0] += -0.028480927051469096;
                } else {
                  result[0] += 0.027162052619255264;
                }
              } else {
                result[0] += 0.06961208360362252;
              }
            } else {
              if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2252.000000000000455) ) ) {
                result[0] += 0.0005153784279816074;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                  result[0] += -0.12586261695606554;
                } else {
                  result[0] += -0.02911737458040556;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
            result[0] += 0.019344709854805124;
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.45023441314697443) ) ) {
              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += 0.04323950768206506;
              } else {
                result[0] += 0.19343841688954377;
              }
            } else {
              result[0] += 0.014428394224445149;
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.801661729812622958) ) ) {
        if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)8816427008.000001907) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.65098142623901545) ) ) {
            result[0] += 0.07603462154660504;
          } else {
            result[0] += 0.0011901433908838488;
          }
        } else {
          if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.534971714019776279) ) ) {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.046559436015007415;
              } else {
                if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)7.694305181503296787) ) ) {
                  result[0] += -0.007511430853892923;
                } else {
                  result[0] += -0.04017637914593464;
                }
              }
            } else {
              result[0] += -0.05613780733668192;
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.142630577087403232) ) ) {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                  result[0] += 0.046681518721523686;
                } else {
                  result[0] += -0.03560465364893136;
                }
              } else {
                result[0] += -0.03649489295673136;
              }
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.158952236175537998) ) ) {
                result[0] += 0.0013753129813682023;
              } else {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.484580039978028232) ) ) {
                    result[0] += -0.097550252386784;
                  } else {
                    result[0] += -0.026054991758109367;
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.694163918495178667) ) ) {
                    result[0] += 0.102916172778456;
                  } else {
                    result[0] += -0.011487039843072464;
                  }
                }
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.827801465988160068) ) ) {
            result[0] += 0.03496081751889702;
          } else {
            result[0] += -0.07322723599307848;
          }
        } else {
          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.172047138214112216) ) ) {
              result[0] += 0.03807552250445924;
            } else {
              result[0] += -0.021077378003298442;
            }
          } else {
            result[0] += -0.0463733834785144;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
        result[0] += -0.001258451281725835;
      } else {
        if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.261864185333252841) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.302512168884278232) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                result[0] += 0.009810095957206208;
              } else {
                result[0] += -0.005974074114410505;
              }
            } else {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                result[0] += -0.007065134714693866;
              } else {
                result[0] += 0.025286802012642492;
              }
            }
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
              result[0] += -0.011360044094872403;
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.172047138214112216) ) ) {
                result[0] += -0.028237621290979193;
              } else {
                result[0] += 0.13091395322482088;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
            if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.154959201812744585) ) ) {
                  result[0] += -0.06514305641506846;
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.088880300521851474) ) ) {
                    result[0] += 0.09100340476447337;
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.208071470260621005) ) ) {
                      result[0] += -0.15239636980911403;
                    } else {
                      result[0] += 0.06932420130858853;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.715336322784424716) ) ) {
                  result[0] += -0.0331591039747148;
                } else {
                  result[0] += 0.04435471192934692;
                }
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.694163918495178667) ) ) {
                result[0] += -0.026621272131564053;
              } else {
                result[0] += -0.07875743097946786;
              }
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.607751369476319248) ) ) {
              if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += -0.07301372980379049;
              } else {
                result[0] += -0.011588358293322125;
              }
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.161602735519410068) ) ) {
                if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                  result[0] += -0.015457812429895402;
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.172047138214112216) ) ) {
                    result[0] += 0.07323434822779493;
                  } else {
                    if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                      result[0] += 0.01242283625346046;
                    } else {
                      result[0] += 0.08065945934790233;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.487163543701172763) ) ) {
                  result[0] += 0.07765790394881507;
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.531007289886475498) ) ) {
                    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
                      result[0] += -0.1111213776928861;
                    } else {
                      result[0] += 0.03915333401207369;
                    }
                  } else {
                    result[0] += 0.042342320032042996;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
      result[0] += 0.06877343005794782;
    } else {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)1.497866153717041238) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.736135363578796831) ) ) {
          result[0] += 0.05006235148194171;
        } else {
          result[0] += 0.0009047926416351967;
        }
      } else {
        if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.760117292404175249) ) ) {
          if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.431901693344116655) ) ) {
            if ( UNLIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.894675970077515537) ) ) {
                result[0] += -0.07563886847962051;
              } else {
                result[0] += -0.28483692119551063;
              }
            } else {
              result[0] += -0.013425361135666919;
            }
          } else {
            result[0] += 0.0034979327651574223;
          }
        } else {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)14.74696540832519709) ) ) {
            if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
              if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)2.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.861792564392090288) ) ) {
                    result[0] += 0.03589449115517845;
                  } else {
                    if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
                      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                        result[0] += -0.12668452727658128;
                      } else {
                        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.067782521247864214) ) ) {
                          result[0] += -0.09306572532459881;
                        } else {
                          result[0] += 0.09319872740751162;
                        }
                      }
                    } else {
                      result[0] += 0.02261933739412824;
                    }
                  }
                } else {
                  result[0] += -0.017367714972510925;
                }
              } else {
                if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.607751369476319248) ) ) {
                    result[0] += -0.02527292746103493;
                  } else {
                    result[0] += 0.08663907935291425;
                  }
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.57544851303100764) ) ) {
                    result[0] += 0.16351857874858416;
                  } else {
                    result[0] += 0.018011520071694275;
                  }
                }
              }
            } else {
              result[0] += -0.0196888209175909;
            }
          } else {
            if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
              result[0] += -0.11827941858395286;
            } else {
              result[0] += -0.00513828019138176;
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.135017871856690341) ) ) {
          if ( UNLIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += -0.1319950462796847;
          } else {
            result[0] += 0.011509020194792214;
          }
        } else {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.261864185333252841) ) ) {
            result[0] += -0.0024969313786030317;
          } else {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += -0.028833854520811098;
            } else {
              if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.478159427642823154) ) ) {
                  result[0] += 0.07491976101120222;
                } else {
                  result[0] += -0.03338496984222163;
                }
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.070054531097412998) ) ) {
                  result[0] += 0.055383468505973504;
                } else {
                  result[0] += -0.009123861779995957;
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.215408444404602495) ) ) {
          result[0] += 0.003989251934409165;
        } else {
          if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += -0.07983484384396305;
          } else {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.134366035461426669) ) ) {
              result[0] += -0.009098430454184143;
            } else {
              result[0] += -0.037918194972843176;
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
          if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.90474271774292081) ) ) {
              if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                  result[0] += 0.011631488018632383;
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)15.02900028228759943) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.45958471298217951) ) ) {
                      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.920601367950440341) ) ) {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
                          result[0] += -0.005995761626411172;
                        } else {
                          result[0] += 0.0316606844676788;
                        }
                      } else {
                        result[0] += -0.10738110345460579;
                      }
                    } else {
                      result[0] += -0.06288010808847484;
                    }
                  } else {
                    result[0] += 0.07986898184272223;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.484580039978028232) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.77496147155761896) ) ) {
                    result[0] += 0.07851532985392332;
                  } else {
                    result[0] += -0.00959672033237891;
                  }
                } else {
                  result[0] += -0.041499938966150694;
                }
              }
            } else {
              if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                result[0] += -0.02073157448958795;
              } else {
                result[0] += -0.09547792255259424;
              }
            }
          } else {
            if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.87254357337951749) ) ) {
                    result[0] += 0.06781267403325444;
                  } else {
                    result[0] += -0.04522166623043211;
                  }
                } else {
                  result[0] += -0.014811354098822952;
                }
              } else {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)6.117118597030640537) ) ) {
                  result[0] += 0.00864287933887873;
                } else {
                  result[0] += -0.10926561505164455;
                }
              }
            } else {
              if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.07976742119720648;
              } else {
                result[0] += 0.028985913817258197;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.552972793579102007) ) ) {
            result[0] += -0.08318639074370493;
          } else {
            if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)4.500000000000000888) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.994492053985595925) ) ) {
                result[0] += 0.059522908472014746;
              } else {
                if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.06530126732492647;
                  } else {
                    result[0] += -0.001218603409328372;
                  }
                } else {
                  result[0] += 0.004994958401569983;
                }
              }
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)15.34107780456543146) ) ) {
                if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
                    if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                      result[0] += -0.042018540947201755;
                    } else {
                      result[0] += -0.00753740527529498;
                    }
                  } else {
                    if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += 0.29086523191187463;
                    } else {
                      result[0] += 0.027157064553178498;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
                    result[0] += 7.610984274154874e-07;
                  } else {
                    result[0] += -0.06895154285743015;
                  }
                }
              } else {
                result[0] += -0.08614289414052106;
              }
            }
          }
        }
      } else {
        result[0] += 0.004289062303689203;
      }
    }
  }
}

