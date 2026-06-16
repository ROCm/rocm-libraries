
#include "header.h"

void predict_unit3(union Entry* data, double* result) {
  unsigned int tmp;
  if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
    if ( UNLIKELY(  (data[63].missing != -1) && (data[63].fvalue <= (double)-1.00000001800250948e-35) ) ) {
      result[0] += 0.09373450578304968;
    } else {
      if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)6.000000000000000888) ) ) {
        result[0] += -0.003732241868233171;
      } else {
        result[0] += -0.00033584449920495463;
      }
    }
  } else {
    if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.337269306182862216) ) ) {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.029068946838379794) ) ) {
        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.384830474853516513) ) ) {
            result[0] += 0.003537231734214065;
          } else {
            result[0] += -0.02438409133378738;
          }
        } else {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.88256025314331232) ) ) {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)102.5000000000000142) ) ) {
              result[0] += -0.027700352699031257;
            } else {
              result[0] += -0.007508159356974745;
            }
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += 0.01033617346442464;
            } else {
              result[0] += -0.03356931206897365;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.029068946838379794) ) ) {
              result[0] += -0.0022566368520554538;
            } else {
              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.53326439857482999) ) ) {
                  result[0] += 0.03932316087565496;
                } else {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.013626912533344715;
                  } else {
                    if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.120439291000367099) ) ) {
                        result[0] += 0.08682260563436614;
                      } else {
                        result[0] += -0.05129843838572389;
                      }
                    } else {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                        if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.349750161170959917) ) ) {
                          result[0] += -0.04040526307910889;
                        } else {
                          result[0] += -0.16756952239986042;
                        }
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.605120182037354404) ) ) {
                          result[0] += 0.07809549187757509;
                        } else {
                          result[0] += -0.01114406496282798;
                        }
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.828941345214844638) ) ) {
                  if ( LIKELY( !(data[60].missing != -1) || (data[60].fvalue <= (double)6.000000000000000888) ) ) {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.088880300521851474) ) ) {
                        result[0] += -0.017835092315477082;
                      } else {
                        result[0] += -0.06047623229926194;
                      }
                    } else {
                      result[0] += 0.01607049890339775;
                    }
                  } else {
                    result[0] += 0.0015342401843990196;
                  }
                } else {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                    if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
                        result[0] += 0.026790744587959676;
                      } else {
                        result[0] += -0.0020768095802711525;
                      }
                    } else {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.601370334625245029) ) ) {
                        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += 0.0035660607367544307;
                        } else {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.587308406829834873) ) ) {
                            result[0] += 0.008311041432651456;
                          } else {
                            result[0] += -0.037347923473275334;
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += -0.03232258839677242;
                        } else {
                          result[0] += 0.02591780479235799;
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)3.921924352645874468) ) ) {
                      if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)271.5000000000000568) ) ) {
                        result[0] += 0.023560477533800143;
                      } else {
                        result[0] += -0.03177144336866;
                      }
                    } else {
                      result[0] += -0.039134310653601886;
                    }
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
              if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)302.5000000000000568) ) ) {
                  result[0] += 0.003910261321780369;
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.617236852645874912) ) ) {
                    result[0] += -0.06793361448336223;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.679712533950806552) ) ) {
                      result[0] += 0.09840349584841057;
                    } else {
                      result[0] += -0.06553162675053596;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.436733961105347568) ) ) {
                  result[0] += -0.029518771109709425;
                } else {
                  result[0] += 0.0024064616741555503;
                }
              }
            } else {
              result[0] += 0.03549535252738034;
            }
          }
        } else {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.238486170768738237) ) ) {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)100.5000000000000142) ) ) {
              result[0] += -0.033017844221520376;
            } else {
              result[0] += 0.003570428902576397;
            }
          } else {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)156.5000000000000284) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.450390577316285068) ) ) {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.725620865821838823) ) ) {
                  result[0] += -0.002860174734932148;
                } else {
                  result[0] += 0.026081757097920585;
                }
              } else {
                if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.049172236212275774;
                } else {
                  result[0] += 0.16902726056839326;
                }
              }
            } else {
              if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += 0.009122278055242988;
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.329314231872559482) ) ) {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.650573849678039995) ) ) {
                    result[0] += 0.030556285343730862;
                  } else {
                    result[0] += -0.026830831323543067;
                  }
                } else {
                  result[0] += -0.100930973198672;
                }
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.040618419647218573) ) ) {
          if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)147.5000000000000284) ) ) {
            result[0] += -0.09440870476963933;
          } else {
            result[0] += -0.00875723237061634;
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.553655147552491123) ) ) {
            if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.32411074638366788) ) ) {
                result[0] += -0.12496699884235563;
              } else {
                result[0] += 0.019316055929308767;
              }
            } else {
              result[0] += -0.022545005161017016;
            }
          } else {
            if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += -0.0042563740392680665;
            } else {
              result[0] += 0.01911283460769287;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.497866153717041238) ) ) {
          if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                result[0] += 0.03288914202405392;
              } else {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
                  result[0] += 0.02020473470912427;
                } else {
                  result[0] += -0.0673308760714369;
                }
              }
            } else {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.0433133461962425;
              } else {
                result[0] += -0.002106545039546179;
              }
            }
          } else {
            result[0] += -0.06648575389644994;
          }
        } else {
          result[0] += 0.02063892695167746;
        }
      }
    }
  }
  if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
    if ( UNLIKELY(  (data[44].missing != -1) && (data[44].fvalue <= (double)-1.00000001800250948e-35) ) ) {
      result[0] += 0.10014523167934704;
    } else {
      result[0] += -0.0005461241593527138;
    }
  } else {
    if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.337269306182862216) ) ) {
      if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.342454433441162998) ) ) {
            result[0] += 0.07395117608867217;
          } else {
            result[0] += -0.013171680330813854;
          }
        } else {
          if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.012675821781158891) ) ) {
            result[0] += -0.030934588767248285;
          } else {
            result[0] += 0.028590333201527252;
          }
        }
      } else {
        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.570234775543213779) ) ) {
          if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)96.50000000000001421) ) ) {
            result[0] += -0.018644703585029742;
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.241523027420044833) ) ) {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.051747083663941318) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.397998809814454013) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.972535848617554599) ) ) {
                        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.524927973747253862) ) ) {
                          result[0] += 0.031532933004492565;
                        } else {
                          result[0] += 0.13309470544605764;
                        }
                      } else {
                        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
                          if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                            result[0] += 0.04640094494691321;
                          } else {
                            if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.03420138359069913) ) ) {
                              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.868834793567657693) ) ) {
                                if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.861792564392090288) ) ) {
                                  result[0] += 0.06408895478679182;
                                } else {
                                  result[0] += -0.02198378127269365;
                                }
                              } else {
                                result[0] += -0.07948123824244602;
                              }
                            } else {
                              result[0] += -0.07337637736591045;
                            }
                          }
                        } else {
                          if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.068990230560303623) ) ) {
                            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.917405366897583452) ) ) {
                              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.7592954635620135) ) ) {
                                result[0] += -0.008769825050309784;
                              } else {
                                result[0] += 0.05674058030677977;
                              }
                            } else {
                              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.349750161170959917) ) ) {
                                result[0] += 0.06935769978107903;
                              } else {
                                if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)0.8958797454833985485) ) ) {
                                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.449861526489258257) ) ) {
                                    result[0] += 0.08879504760344084;
                                  } else {
                                    result[0] += 0.007384105632082915;
                                  }
                                } else {
                                  result[0] += -0.05864343554269044;
                                }
                              }
                            }
                          } else {
                            result[0] += -0.022431288988828506;
                          }
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.637949228286744052) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.213027238845826083) ) ) {
                          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)300.5000000000000568) ) ) {
                            result[0] += -0.008288309430039097;
                          } else {
                            result[0] += -0.21545111084266197;
                          }
                        } else {
                          result[0] += -0.030271898252910434;
                        }
                      } else {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.835998296737671787) ) ) {
                          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)296.5000000000000568) ) ) {
                            result[0] += -0.003527201325110604;
                          } else {
                            result[0] += 0.06829904937378199;
                          }
                        } else {
                          result[0] += -0.01897310616091545;
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.835998296737671787) ) ) {
                      result[0] += 0.01101126661425981;
                    } else {
                      result[0] += 0.07346131341279566;
                    }
                  }
                } else {
                  result[0] += 0.008608512425704045;
                }
              } else {
                result[0] += -0.007199027190696886;
              }
            } else {
              if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)7.223295450210572177) ) ) {
                if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.553655147552491123) ) ) {
                    if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)48.00000000000000711) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.040716171264650214) ) ) {
                        result[0] += 0.017397027232319128;
                      } else {
                        result[0] += -0.02244784763497372;
                      }
                    } else {
                      if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.31402075290679976) ) ) {
                        result[0] += -0.11032905981174007;
                      } else {
                        result[0] += 0.014207534925136895;
                      }
                    }
                  } else {
                    result[0] += -6.836976037122264e-08;
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.649621725082398349) ) ) {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                      result[0] += -0.018699017767356826;
                    } else {
                      result[0] += 0.016195722785546648;
                    }
                  } else {
                    if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.497866153717041238) ) ) {
                      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                        result[0] += 0.008109909492849668;
                      } else {
                        if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)189.5000000000000284) ) ) {
                          result[0] += -0.007794537523869588;
                        } else {
                          result[0] += -0.06351728465934524;
                        }
                      }
                    } else {
                      result[0] += 0.03569460522419945;
                    }
                  }
                }
              } else {
                result[0] += 0.009078793341474508;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.192109584808350498) ) ) {
              result[0] += 0.011776626944404647;
            } else {
              result[0] += 0.036397643997380984;
            }
          } else {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)159.5000000000000284) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.01535444424549993;
              } else {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += 0.038471237975030215;
                } else {
                  result[0] += -0.00723146980430075;
                }
              }
            } else {
              result[0] += -0.014895582759432336;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.131699204444885698) ) ) {
        result[0] += -0.04239414516421187;
      } else {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.861792564392090288) ) ) {
            result[0] += 0.0033395643440475697;
          } else {
            if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)10.2051577568054217) ) ) {
                result[0] += -0.014214765711423598;
              } else {
                result[0] += 0.07157155378300643;
              }
            } else {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += 0.012109206339014147;
              } else {
                result[0] += 0.035178588421113384;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)192.5000000000000284) ) ) {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
              result[0] += 0.01716117334486718;
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.241523027420044833) ) ) {
                result[0] += -0.0530671165291763;
              } else {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.024254913954280047;
                } else {
                  result[0] += -0.007226838080720894;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.497866153717041238) ) ) {
              if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)238.5000000000000284) ) ) {
                  result[0] += -0.05053507027292005;
                } else {
                  result[0] += 0.01892434563906701;
                }
              } else {
                result[0] += -0.026213412718626317;
              }
            } else {
              result[0] += 0.015389412606761792;
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
    result[0] += -0.0005620261986091316;
  } else {
    if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.166635274887085849) ) ) {
      if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
        result[0] += 0.015692245307407024;
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.029068946838379794) ) ) {
          if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)104.5000000000000142) ) ) {
            result[0] += -0.019473365600046144;
          } else {
            if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.51693725585937678) ) ) {
                if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)279.5000000000000568) ) ) {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.241523027420044833) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.989220380783081943) ) ) {
                        result[0] += 0.02181668223456515;
                      } else {
                        if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                          if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
                            result[0] += -0.02326913850425014;
                          } else {
                            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.174569487571716753) ) ) {
                              result[0] += 0.004372312148018986;
                            } else {
                              if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                                result[0] += 0.03492066743859622;
                              } else {
                                result[0] += -0.057952796994724345;
                              }
                            }
                          }
                        } else {
                          result[0] += 0.0180059451648944;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.686429500579835761) ) ) {
                        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.501469135284425604) ) ) {
                          result[0] += -0.02248111353394891;
                        } else {
                          result[0] += -0.07579108292907893;
                        }
                      } else {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.69067406654357999) ) ) {
                          result[0] += -0.029811610324078603;
                        } else {
                          if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                            result[0] += -0.03438690620230302;
                          } else {
                            result[0] += 0.00333899512448781;
                          }
                        }
                      }
                    }
                  } else {
                    result[0] += 0.0015638372061908346;
                  }
                } else {
                  result[0] += 0.003993080132069562;
                }
              } else {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                  result[0] += 0.012595779895195525;
                } else {
                  result[0] += -0.019224953017541704;
                }
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.025192260742188388) ) ) {
                result[0] += 0.07118063265245465;
              } else {
                result[0] += 0.011570634853191326;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.029068946838379794) ) ) {
                result[0] += -0.0021437111173692723;
              } else {
                result[0] += 0.003469005637513486;
              }
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += -0.002357940529897612;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.512576580047609198) ) ) {
                    if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)147.5000000000000284) ) ) {
                      result[0] += -0.05566960174132515;
                    } else {
                      result[0] += -0.019239400936653772;
                    }
                  } else {
                    result[0] += -0.004881245374727115;
                  }
                }
              } else {
                result[0] += 0.032790584097335575;
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.587308406829834873) ) ) {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.029023735924207796;
              } else {
                if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                    result[0] += -0.004196495184983154;
                  } else {
                    if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                        result[0] += 0.01467347471246876;
                      } else {
                        result[0] += -0.11357172545242827;
                      }
                    } else {
                      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.537586927413940874) ) ) {
                        result[0] += 0.029785832731971437;
                      } else {
                        result[0] += 0.10703283735095277;
                      }
                    }
                  }
                } else {
                  result[0] += 0.042864178245737745;
                }
              }
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.024269006446385383;
                } else {
                  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += -0.005280429474304755;
                  } else {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.088880300521851474) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.61636352539062678) ) ) {
                        if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                          result[0] += 0.018970524923134245;
                        } else {
                          result[0] += -0.01486839512030652;
                        }
                      } else {
                        result[0] += 0.02064906188922722;
                      }
                    } else {
                      if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                        result[0] += 0.002614771500379843;
                      } else {
                        result[0] += 0.037087825239286866;
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                    result[0] += -0.017425558806148968;
                  } else {
                    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                      result[0] += 0.06032405949220267;
                    } else {
                      result[0] += -0.005090991292929035;
                    }
                  }
                } else {
                  result[0] += -0.05533319707221906;
                }
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.342454433441162998) ) ) {
        if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
          if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.836270570755005771) ) ) {
            result[0] += 0.008093181221953584;
          } else {
            result[0] += 0.05368077130447072;
          }
        } else {
          result[0] += -0.018389560291687194;
        }
      } else {
        if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
          result[0] += 0.00041610061591748865;
        } else {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.126503944396974433) ) ) {
              if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.700598716735840066) ) ) {
                result[0] += 0.011598806497677885;
              } else {
                result[0] += -0.12342296180171104;
              }
            } else {
              if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.321723937988282138) ) ) {
                    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
                        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                          result[0] += 0.027386734329004736;
                        } else {
                          result[0] += -0.03121270939501272;
                        }
                      } else {
                        result[0] += 0.04358120409369042;
                      }
                    } else {
                      if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)234.5000000000000284) ) ) {
                        if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)144.5000000000000284) ) ) {
                          result[0] += -0.035522058376276054;
                        } else {
                          result[0] += 0.02000104723652314;
                        }
                      } else {
                        result[0] += -0.05670795611391595;
                      }
                    }
                  } else {
                    result[0] += 0.026631224713103946;
                  }
                } else {
                  result[0] += -0.06357391553046243;
                }
              } else {
                result[0] += 0.022778711248266418;
              }
            }
          } else {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.962127923965454546) ) ) {
                result[0] += -0.012408684638213666;
              } else {
                result[0] += 0.04527577471780576;
              }
            } else {
              result[0] += -0.03750175450859197;
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[58].missing != -1) || (data[58].fvalue <= (double)1.500000000000000222) ) ) {
    if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)319.5000000000000568) ) ) {
      if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.64616632461548029) ) ) {
          if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)3.500000000000000444) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.921100616455079013) ) ) {
                result[0] += 0.0029491232068432124;
              } else {
                result[0] += -0.06113742255376801;
              }
            } else {
              result[0] += 0.04239908552472003;
            }
          } else {
            if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.059444409039138946;
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.579273939132691318) ) ) {
                if ( UNLIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.126503944396974433) ) ) {
                    if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)64.50000000000001421) ) ) {
                      result[0] += -0.05869998630970833;
                    } else {
                      result[0] += -0.007855712162860058;
                    }
                  } else {
                    result[0] += 0.005019031480685767;
                  }
                } else {
                  if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5.000579357147217685) ) ) {
                    result[0] += 0.0096861177363281;
                  } else {
                    result[0] += -0.03627809222453183;
                  }
                }
              } else {
                if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)2.500000000000000444) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.97070193290710538) ) ) {
                    result[0] += 0.0031021075847972456;
                  } else {
                    if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.05711799551867269;
                    } else {
                      result[0] += 0.003980235205547911;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.426736354827881748) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.363266706466675693) ) ) {
                      result[0] += -0.009359484974230682;
                    } else {
                      result[0] += -0.06377140814812493;
                    }
                  } else {
                    result[0] += 0.03622941558606543;
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)6.000000000000000888) ) ) {
              if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                result[0] += 0.026642753127325194;
              } else {
                result[0] += 0.003925236160722433;
              }
            } else {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)87.50000000000001421) ) ) {
                result[0] += -0.007039467299351995;
              } else {
                result[0] += -0.03303451509440394;
              }
            }
          } else {
            result[0] += -0.03742586504767189;
          }
        }
      } else {
        if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)0.8958797454833985485) ) ) {
          result[0] += -0.013141463976333019;
        } else {
          if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
            if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.744487762451173651) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.182065486907959873) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.070172666736035;
                  } else {
                    result[0] += 0.003460081524411471;
                  }
                } else {
                  result[0] += -0.022946988053666222;
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.553712725639343706) ) ) {
                  result[0] += -0.003242929870553784;
                } else {
                  if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.028988208064274025;
                  } else {
                    result[0] += -0.06514488626887312;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.775349855422974521) ) ) {
                if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)21.50000000000000355) ) ) {
                    result[0] += -0.05066901827743689;
                  } else {
                    result[0] += -0.012204496435119215;
                  }
                } else {
                  result[0] += 0.006038518093107845;
                }
              } else {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.322819471359253818) ) ) {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
                        result[0] += -0.02423533107299138;
                      } else {
                        result[0] += 0.010901237347004576;
                      }
                    } else {
                      result[0] += -0.03069308404884475;
                    }
                  } else {
                    result[0] += 0.002573656164871945;
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.262283086776734287) ) ) {
                    result[0] += -0.02737805348267698;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.420236110687257636) ) ) {
                      if ( UNLIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)2.500000000000000444) ) ) {
                        result[0] += -0.009782334687614231;
                      } else {
                        result[0] += 0.02017566367852601;
                      }
                    } else {
                      result[0] += 0.030321368907449436;
                    }
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.662244915962219682) ) ) {
                if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.497866153717041238) ) ) {
                    result[0] += -0.02050634510967052;
                  } else {
                    if ( UNLIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.030515278218668236;
                    } else {
                      if ( UNLIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.737386107444763628) ) ) {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.61675357818603693) ) ) {
                            result[0] += -0.0052329449495423265;
                          } else {
                            result[0] += 0.05708280922702083;
                          }
                        } else {
                          result[0] += -0.03469996768575024;
                        }
                      } else {
                        if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
                          result[0] += -0.015181998935839115;
                        } else {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.43749904632568537) ) ) {
                            result[0] += -0.004793987344260149;
                          } else {
                            result[0] += 0.057774181991077124;
                          }
                        }
                      }
                    }
                  }
                } else {
                  result[0] += -0.017153218821331404;
                }
              } else {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)32.50000000000000711) ) ) {
                  result[0] += 0.00506301684094986;
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.79402017593383967) ) ) {
                    if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.021842018486925022;
                    } else {
                      result[0] += 0.00200040638544835;
                    }
                  } else {
                    if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                      result[0] += -0.009109250924487234;
                    } else {
                      result[0] += -0.050670097568298905;
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.216319084167481357) ) ) {
                result[0] += -0.005387355283297279;
              } else {
                result[0] += 0.004430386081890447;
              }
            }
          }
        }
      }
    } else {
      result[0] += -0.016397547082507718;
    }
  } else {
    if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
      if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)4.500000000000000888) ) ) {
        result[0] += 7.922615727547665e-05;
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.272946834564209873) ) ) {
          result[0] += 0.009505305576326933;
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.216319084167481357) ) ) {
            result[0] += 0.02545676752142432;
          } else {
            result[0] += -0.07678925699329452;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.828941345214844638) ) ) {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
          result[0] += 0.08021419406680051;
        } else {
          if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)2.500000000000000444) ) ) {
            result[0] += -0.02346655661144395;
          } else {
            result[0] += 0.001971027559615668;
          }
        }
      } else {
        result[0] += 0.00385795502544916;
      }
    }
  }
  if ( LIKELY( !(data[58].missing != -1) || (data[58].fvalue <= (double)1.500000000000000222) ) ) {
    if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)319.5000000000000568) ) ) {
      if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
          if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.851041555404663974) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.842459201812745917) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.500490188598633701) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
                    result[0] += 0.006340498209252194;
                  } else {
                    if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                      result[0] += -0.015671473859304747;
                    } else {
                      result[0] += -0.058872057201868604;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                    if ( UNLIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += -0.04570543203627087;
                    } else {
                      result[0] += 0.015211587053604337;
                    }
                  } else {
                    result[0] += 0.08075315020926017;
                  }
                }
              } else {
                if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                  if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)65.50000000000001421) ) ) {
                    if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.05323502135470413;
                    } else {
                      if ( UNLIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                        if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                          result[0] += -0.06625739050313166;
                        } else {
                          result[0] += 0.0201433102401021;
                        }
                      } else {
                        if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                          result[0] += 0.055858078962170515;
                        } else {
                          result[0] += -0.060593681762920126;
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                      result[0] += 0.017477639859091048;
                    } else {
                      result[0] += -0.021611816776364808;
                    }
                  }
                } else {
                  result[0] += -0.011029118082660718;
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
                result[0] += -0.09795158582701852;
              } else {
                result[0] += -0.03202092668467489;
              }
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.21334457397461115) ) ) {
              result[0] += 0.0060461796810734785;
            } else {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.241523027420044833) ) ) {
                  result[0] += -0.04692615508987144;
                } else {
                  result[0] += -0.004003459545902855;
                }
              } else {
                if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += 0.06811553670483175;
                } else {
                  result[0] += -0.02400650036923133;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.216319084167481357) ) ) {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)68.50000000000001421) ) ) {
              result[0] += -0.006355627145330875;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.88024568557739435) ) ) {
                result[0] += 0.006273331607215821;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.802696108818054643) ) ) {
                  if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)83.50000000000001421) ) ) {
                    result[0] += 0.03480549728044521;
                  } else {
                    result[0] += -0.013660274811437563;
                  }
                } else {
                  result[0] += -0.018171461994310984;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)56.50000000000000711) ) ) {
              result[0] += 0.018086165367269365;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.744487762451173651) ) ) {
                result[0] += -0.0028652163162298534;
              } else {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)66.50000000000001421) ) ) {
                  result[0] += -0.0020773827501239504;
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.801954269409180576) ) ) {
                    result[0] += -0.01285568136518557;
                  } else {
                    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.0203235447370344;
                    } else {
                      result[0] += -0.06981682952961762;
                    }
                  }
                }
              }
            }
          }
        }
      } else {
        result[0] += -0.0015041281740916731;
      }
    } else {
      result[0] += -0.015055929149615621;
    }
  } else {
    if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
      if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
          if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)248.5000000000000284) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.53326439857482999) ) ) {
                  result[0] += 0.0404492222928589;
                } else {
                  result[0] += -0.024919561820112457;
                }
              } else {
                result[0] += 0.059464481380285594;
              }
            } else {
              if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)228.5000000000000284) ) ) {
                result[0] += 0.03969817538948191;
              } else {
                result[0] += -0.01894103194329789;
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.158010244369507724) ) ) {
              result[0] += -0.11749190503028391;
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.623839378356934482) ) ) {
                result[0] += -0.03992903256351543;
              } else {
                result[0] += 0.028565269393658874;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
            if ( UNLIKELY( !(data[56].missing != -1) || (data[56].fvalue <= (double)3.000000000000000444) ) ) {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)286.5000000000000568) ) ) {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)172.5000000000000284) ) ) {
                  result[0] += -0.0022809863262000882;
                } else {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.016823921897125862;
                  } else {
                    result[0] += -0.010510791450379052;
                  }
                }
              } else {
                result[0] += -0.02159404835327385;
              }
            } else {
              if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.012675821781158891) ) ) {
                if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)300.5000000000000568) ) ) {
                  result[0] += 0.016435120065551416;
                } else {
                  result[0] += -0.02152017075052269;
                }
              } else {
                result[0] += -0.004214872755995489;
              }
            }
          } else {
            result[0] += 0.001383642315178609;
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.439304351806642401) ) ) {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
            result[0] += 0.0035502868200059535;
          } else {
            result[0] += -0.03281771301309431;
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.397998809814454013) ) ) {
            if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.602003335952759233) ) ) {
              result[0] += 0.4762578047315413;
            } else {
              result[0] += -0.009012306438475474;
            }
          } else {
            result[0] += -0.06861161383500973;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.828941345214844638) ) ) {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.835998296737671787) ) ) {
            result[0] += 0.048324960325710514;
          } else {
            result[0] += -0.016576257414930767;
          }
        } else {
          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.921924352645874468) ) ) {
              result[0] += 0.07889805879459111;
            } else {
              result[0] += -0.003866678427002992;
            }
          } else {
            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
              if ( UNLIKELY( !(data[60].missing != -1) || (data[60].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += -0.04955978182654643;
              } else {
                result[0] += 0.028970215548368334;
              }
            } else {
              result[0] += -0.09271478445290059;
            }
          }
        }
      } else {
        result[0] += 0.00370089509657329;
      }
    }
  }
  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.556798219680787021) ) ) {
      if ( UNLIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)2.500000000000000444) ) ) {
        if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)66.50000000000001421) ) ) {
            if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)48.50000000000000711) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.497866153717041238) ) ) {
                result[0] += 0.10119254954603352;
              } else {
                result[0] += -0.015208694978441224;
              }
            } else {
              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.497866153717041238) ) ) {
                result[0] += 0.09260831502394537;
              } else {
                result[0] += -0.003388832820202914;
              }
            }
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)1.242453336715698464) ) ) {
              result[0] += -0.1966053759801274;
            } else {
              result[0] += -0.005785601799634481;
            }
          }
        } else {
          if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
            result[0] += -0.06567558082151584;
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.673553824424744096) ) ) {
              result[0] += 0.1452747764653154;
            } else {
              result[0] += -0.021411672207014475;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.835998296737671787) ) ) {
          if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
            result[0] += 0.08457032406259189;
          } else {
            result[0] += 0.02890571858737397;
          }
        } else {
          if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)4.500000000000000888) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.745876312255860263) ) ) {
              result[0] += -0.031643770296349925;
            } else {
              result[0] += 0.023184598001053203;
            }
          } else {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.947818994522095615) ) ) {
              result[0] += 0.028286299180758596;
            } else {
              result[0] += -0.03128817645507824;
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.182065486907959873) ) ) {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.88256025314331232) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.778982400894165927) ) ) {
            if ( UNLIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.79084348678589045) ) ) {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.481121778488159624) ) ) {
                  result[0] += -0.0469751735938518;
                } else {
                  if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)47.50000000000000711) ) ) {
                    if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.350240230560303178) ) ) {
                      result[0] += 0.04530416548026908;
                    } else {
                      result[0] += -0.038225176861429955;
                    }
                  } else {
                    result[0] += -0.008669686018573488;
                  }
                }
              } else {
                result[0] += 0.028228727383782116;
              }
            } else {
              result[0] += 0.02965888531047909;
            }
          } else {
            if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)4.500000000000000888) ) ) {
              if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.040618419647218573) ) ) {
                    result[0] += -0.0016905674516961525;
                  } else {
                    result[0] += -0.02053201598367209;
                  }
                } else {
                  if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.020127415657043901) ) ) {
                    result[0] += 0.01953048082778631;
                  } else {
                    if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.518409252166748935) ) ) {
                      if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.470024824142456943) ) ) {
                        if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.238486170768738237) ) ) {
                          if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.379274368286133701) ) ) {
                            if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.156774044036865678) ) ) {
                              result[0] += 0.0058780965657707004;
                            } else {
                              result[0] += -0.10549279585337588;
                            }
                          } else {
                            result[0] += 0.08673219948057576;
                          }
                        } else {
                          result[0] += -0.09614253648655047;
                        }
                      } else {
                        result[0] += -0.12056868269590536;
                      }
                    } else {
                      if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.540307998657227451) ) ) {
                        result[0] += 0.17545461761311862;
                      } else {
                        result[0] += 0.004051299664874517;
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.649621725082398349) ) ) {
                  if ( UNLIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.053812814514614594;
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.737386107444763628) ) ) {
                      result[0] += 0.030889816224146785;
                    } else {
                      result[0] += -0.017284470328290186;
                    }
                  }
                } else {
                  result[0] += -0.018321193662775095;
                }
              }
            } else {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.637949228286744052) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                  result[0] += 0.031522972276242955;
                } else {
                  result[0] += -0.06511084787799691;
                }
              } else {
                result[0] += -0.001956541078143636;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)1.242453336715698464) ) ) {
              result[0] += 0.07677430931915097;
            } else {
              result[0] += -0.00927522393145839;
            }
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.142630577087403232) ) ) {
              result[0] += -0.019357585156483656;
            } else {
              result[0] += -0.05586940991309475;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)47.50000000000000711) ) ) {
          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)36.50000000000000711) ) ) {
            result[0] += -0.05273734007846806;
          } else {
            result[0] += 0.019752243664387045;
          }
        } else {
          if ( UNLIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)2.500000000000000444) ) ) {
            result[0] += -0.060956306816856445;
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
              if ( UNLIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)4.500000000000000888) ) ) {
                result[0] += 0.028827814916190294;
              } else {
                result[0] += -0.053973904963489944;
              }
            } else {
              result[0] += -0.037541152853447224;
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)6.000000000000000888) ) ) {
      if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.827472925186158115) ) ) {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.10577535629272639) ) ) {
          if ( UNLIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.23832273483276456) ) ) {
              if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.007176153960872199;
              } else {
                if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)9.069797992706300604) ) ) {
                  result[0] += -0.05398043802813285;
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.094205617904663974) ) ) {
                    result[0] += 0.05216025510151143;
                  } else {
                    result[0] += -0.10074993922612883;
                  }
                }
              }
            } else {
              result[0] += -0.06734987990599627;
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.744781017303467685) ) ) {
              result[0] += 0.037967608943649665;
            } else {
              result[0] += -0.0060933710522645375;
            }
          }
        } else {
          if ( UNLIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += -0.014782122187180183;
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.636499762535095659) ) ) {
                if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5.173316955566407138) ) ) {
                    result[0] += 0.04171687694160678;
                  } else {
                    result[0] += 0.1238522656147147;
                  }
                } else {
                  result[0] += -0.08709708542148546;
                }
              } else {
                result[0] += 0.01405676923256933;
              }
            }
          } else {
            result[0] += -0.027677546712468182;
          }
        }
      } else {
        result[0] += -0.06192152725758461;
      }
    } else {
      result[0] += 0.00019582719209722016;
    }
  }
  if ( UNLIKELY(  (data[63].missing != -1) && (data[63].fvalue <= (double)-1.00000001800250948e-35) ) ) {
    result[0] += 0.09513282634426486;
  } else {
    if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)25.50000000000000355) ) ) {
      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.13022470474243342) ) ) {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.285887241363526279) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.338562726974488193) ) ) {
                result[0] += -0.014135659725057433;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.637949228286744052) ) ) {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.006981690829843229;
                  } else {
                    result[0] += -0.07220016576462604;
                  }
                } else {
                  if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)24.50000000000000355) ) ) {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.572941064834595615) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.040716171264650214) ) ) {
                        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += -0.030962321816655303;
                        } else {
                          result[0] += 0.02393161886655273;
                        }
                      } else {
                        result[0] += 0.015068793026849384;
                      }
                    } else {
                      result[0] += 0.045770941617373696;
                    }
                  } else {
                    result[0] += -0.02246414406965444;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.05883158932209218;
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.711936950683595526) ) ) {
                  result[0] += -0.03817333860492328;
                } else {
                  if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)23.50000000000000355) ) ) {
                    result[0] += -0.023214708973988522;
                  } else {
                    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                      result[0] += 0.08148060977126106;
                    } else {
                      result[0] += -0.030334589361828253;
                    }
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.892816066741945136) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.617236852645874912) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.553712725639343706) ) ) {
                  result[0] += -0.0020875634104876497;
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.501469135284425604) ) ) {
                    if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)23.50000000000000355) ) ) {
                      result[0] += -0.0794093960951132;
                    } else {
                      result[0] += -0.024435632433521356;
                    }
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.737386107444763628) ) ) {
                      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += -0.09908804145272665;
                      } else {
                        result[0] += 0.044118237379285254;
                      }
                    } else {
                      result[0] += -0.058013492503901434;
                    }
                  }
                }
              } else {
                result[0] += 0.003560158395045006;
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.737386107444763628) ) ) {
                result[0] += 0.026926071030621057;
              } else {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.56941866874694913) ) ) {
                    result[0] += -0.06025938293352489;
                  } else {
                    result[0] += 0.012863594909016455;
                  }
                } else {
                  result[0] += -0.0033705380740955216;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.464467763900757724) ) ) {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)12.50000000000000178) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.184114694595337802) ) ) {
                result[0] += 0.02124559230987537;
              } else {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.040716171264650214) ) ) {
                    result[0] += 0.015589658105165664;
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
                      result[0] += -0.07086331834967749;
                    } else {
                      result[0] += 0.012095942154213325;
                    }
                  }
                } else {
                  result[0] += -0.0579188963625947;
                }
              }
            } else {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.272946834564209873) ) ) {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)5.500000000000000888) ) ) {
                    if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                      result[0] += 0.012956095711208796;
                    } else {
                      result[0] += -0.05342858949598002;
                    }
                  } else {
                    result[0] += 0.026810214174483968;
                  }
                } else {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.158010244369507724) ) ) {
                      result[0] += 0.021504734956301563;
                    } else {
                      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
                        result[0] += -0.011226529207718133;
                      } else {
                        result[0] += -0.06470815550905405;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.474771499633789951) ) ) {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.56941866874694913) ) ) {
                          result[0] += -0.013970302986942027;
                        } else {
                          result[0] += -0.07617658615198998;
                        }
                      } else {
                        if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
                          result[0] += -0.003868210399323241;
                        } else {
                          result[0] += 0.030038248307061484;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)5.500000000000000888) ) ) {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.637949228286744052) ) ) {
                          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.53326439857482999) ) ) {
                            result[0] += 0.03208300897684817;
                          } else {
                            result[0] += -0.031642930733143934;
                          }
                        } else {
                          if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)6.000000000000000888) ) ) {
                            result[0] += 0.013036084949266469;
                          } else {
                            result[0] += 0.05096844661781052;
                          }
                        }
                      } else {
                        result[0] += -0.01543972452532174;
                      }
                    }
                  }
                }
              } else {
                result[0] += -0.020113356227530518;
              }
            }
          } else {
            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.020127415657043901) ) ) {
              result[0] += -0.007726466349078947;
            } else {
              result[0] += -0.05430935350803884;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.342454433441162998) ) ) {
          if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
            result[0] += 0.0021947630800946013;
          } else {
            result[0] += -0.051281177999543605;
          }
        } else {
          if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += -0.042735443494131255;
            } else {
              if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)7.500000000000000888) ) ) {
                result[0] += 0.030134857647348474;
              } else {
                result[0] += -0.01490325208148802;
              }
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.2121162414550799) ) ) {
              if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.008738896302155059;
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
                    result[0] += -0.06239802411762002;
                  } else {
                    result[0] += -0.0024551148134405215;
                  }
                }
              } else {
                result[0] += 0.027719764536482717;
              }
            } else {
              if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += 0.017531770806354353;
              } else {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.856657028198243964) ) ) {
                    if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.212100267410279208) ) ) {
                      result[0] += 0.07700886578793101;
                    } else {
                      result[0] += -0.09930116985421904;
                    }
                  } else {
                    result[0] += 0.12766594191851907;
                  }
                } else {
                  result[0] += 0.02800059125332015;
                }
              }
            }
          }
        }
      }
    } else {
      result[0] += 0.00015552447043825585;
    }
  }
  if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.00000001800250948e-35) ) ) {
        result[0] += 0.09189578734912912;
      } else {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)6.500000000000000888) ) ) {
          result[0] += -0.0003567772488069965;
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.972535848617554599) ) ) {
            result[0] += 0.007561856956605781;
          } else {
            result[0] += -0.07348801117460349;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.94957673549652144) ) ) {
        if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)6.000000000000000888) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.171656608581543857) ) ) {
            result[0] += 0.004632333584624392;
          } else {
            if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.722943305969239169) ) ) {
              result[0] += 0.06826618069077528;
            } else {
              result[0] += -0.029322100375977978;
            }
          }
        } else {
          result[0] += -0.013095263338928943;
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.744487762451173651) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.993164777755738193) ) ) {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                  result[0] += 0.009062913549729885;
                } else {
                  result[0] += -0.02119055434184193;
                }
              } else {
                result[0] += 0.02157237539334741;
              }
            } else {
              result[0] += -0.013911491475465797;
            }
          } else {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.534971714019776279) ) ) {
              result[0] += -0.014117677013652226;
            } else {
              result[0] += -0.0491065602477925;
            }
          }
        } else {
          if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.310776710510254794) ) ) {
            if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.700598716735840066) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.51693725585937678) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.493027687072754794) ) ) {
                  if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.011420111402365117;
                    } else {
                      result[0] += -0.03869095561225283;
                    }
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.142630577087403232) ) ) {
                      result[0] += -0.018869448167799882;
                    } else {
                      result[0] += 0.10415521892718678;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += -0.05879020486679146;
                  } else {
                    if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.061855646451536954;
                    } else {
                      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
                        result[0] += 0.033805364781583566;
                      } else {
                        result[0] += -0.10028189362021045;
                      }
                    }
                  }
                }
              } else {
                result[0] += -0.05595927704778269;
              }
            } else {
              if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)164.5000000000000284) ) ) {
                result[0] += 0.06915198604889274;
              } else {
                result[0] += -0.028122769640009898;
              }
            }
          } else {
            result[0] += -0.06993574636073244;
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.449861526489258257) ) ) {
      result[0] += -0.0002433071279199437;
    } else {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.53326439857482999) ) ) {
        if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)98.50000000000001421) ) ) {
          result[0] += -0.03898046404198448;
        } else {
          result[0] += 0.0007724170252850453;
        }
      } else {
        if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.745876312255860263) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.96495962142944514) ) ) {
              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.13022470474243342) ) ) {
                  if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
                      result[0] += -0.01707479963995626;
                    } else {
                      result[0] += 0.10052008186471406;
                    }
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.815814018249513495) ) ) {
                      result[0] += 0.00014075480624680024;
                    } else {
                      result[0] += -0.13010099213255796;
                    }
                  }
                } else {
                  result[0] += 0.07529156418624929;
                }
              } else {
                if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += 0.006491082949774559;
                } else {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.637949228286744052) ) ) {
                    result[0] += -0.0043170951499636265;
                  } else {
                    result[0] += -0.040894712396178506;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += 0.000821564498468288;
              } else {
                result[0] += 0.033031136668262906;
              }
            }
          } else {
            if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.868834793567657693) ) ) {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += 0.009883392952663344;
              } else {
                if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.003393893374167683;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.055311203002930576) ) ) {
                    result[0] += -0.14955959838218655;
                  } else {
                    result[0] += 0.036567739952778004;
                  }
                }
              }
            } else {
              result[0] += -0.12955125274910856;
            }
          }
        } else {
          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.780892848968506748) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.764287948608400214) ) ) {
                  result[0] += 0.0317628076635458;
                } else {
                  if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)48.00000000000000711) ) ) {
                    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                      result[0] += -0.000932400368688369;
                    } else {
                      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += -0.023461967062151548;
                      } else {
                        result[0] += -0.06316643715715292;
                      }
                    }
                  } else {
                    result[0] += 0.027947398443208343;
                  }
                }
              } else {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)8.285748958587648261) ) ) {
                    if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)195.5000000000000284) ) ) {
                      result[0] += 0.024584870810168358;
                    } else {
                      if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                        result[0] += 0.034780446897226786;
                      } else {
                        result[0] += -0.01612179119947569;
                      }
                    }
                  } else {
                    result[0] += 0.056224849907980205;
                  }
                } else {
                  result[0] += -0.07463746451121882;
                }
              }
            } else {
              if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)144.5000000000000284) ) ) {
                result[0] += 0.035645612596104465;
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.909855604171753818) ) ) {
                  result[0] += 0.0007495022200368168;
                } else {
                  result[0] += 0.023716236730577024;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.558241367340089667) ) ) {
                  result[0] += 0.0040164702100149744;
                } else {
                  if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
                    result[0] += -0.038529379813194466;
                  } else {
                    result[0] += 0.006287122311694791;
                  }
                }
              } else {
                result[0] += 0.009752603063496582;
              }
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += 0.016083533962833404;
              } else {
                result[0] += -0.06814333330326189;
              }
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[58].missing != -1) || (data[58].fvalue <= (double)1.500000000000000222) ) ) {
    result[0] += -0.0005914199167290158;
  } else {
    if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
      if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)186.5000000000000284) ) ) {
        if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)0.8958797454833985485) ) ) {
          result[0] += -0.031284842816200725;
        } else {
          if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.497866153717041238) ) ) {
            if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.015108142921449748;
            } else {
              if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                result[0] += 0.021370536260936657;
              } else {
                result[0] += -0.09151533888047374;
              }
            }
          } else {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.192109584808350498) ) ) {
              result[0] += -0.004969183256543639;
            } else {
              result[0] += 0.003991237119173701;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)244.5000000000000284) ) ) {
              result[0] += -0.0074932809682923054;
            } else {
              result[0] += 0.05134658170316019;
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.49584054946899592) ) ) {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.288254261016846591) ) ) {
                    result[0] += -0.019456088433423037;
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.102759599685669833) ) ) {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                        if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                          result[0] += 0.006072673704854422;
                        } else {
                          result[0] += -0.08194033603675861;
                        }
                      } else {
                        result[0] += 0.0049356023622265065;
                      }
                    } else {
                      result[0] += -0.04544811423642697;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.088880300521851474) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.288254261016846591) ) ) {
                      if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                        result[0] += 0.06860958081523134;
                      } else {
                        result[0] += -0.042748167892117606;
                      }
                    } else {
                      result[0] += -0.024137959916399557;
                    }
                  } else {
                    result[0] += -0.06868292959392153;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.700598716735840066) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.426736354827881748) ) ) {
                    result[0] += 0.026530630867417533;
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.846404790878296787) ) ) {
                      result[0] += -0.09528108647093136;
                    } else {
                      result[0] += 0.022024082296175723;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)6.000000000000000888) ) ) {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.067782521247864214) ) ) {
                      result[0] += 0.0250014849394496;
                    } else {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.53326439857482999) ) ) {
                        result[0] += -0.02284340474650754;
                      } else {
                        result[0] += 0.011046762310213905;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.289595603942871982) ) ) {
                      result[0] += 9.674586235900011e-05;
                    } else {
                      if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += -0.0038945108316685868;
                      } else {
                        if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.802901029586792436) ) ) {
                          result[0] += 0.01253474872453286;
                        } else {
                          result[0] += 0.0579436226728397;
                        }
                      }
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)324.5000000000000568) ) ) {
                result[0] += 0.014268607454660909;
              } else {
                result[0] += 0.059538163524822954;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)233.5000000000000284) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.192109584808350498) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.329314231872559482) ) ) {
                result[0] += 0.006315889403180521;
              } else {
                result[0] += 0.019184624096035205;
              }
            } else {
              result[0] += -0.0036700237630501657;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.558241367340089667) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.029068946838379794) ) ) {
                if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                  result[0] += -0.04285553265865869;
                } else {
                  if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)2.012675821781158891) ) ) {
                      result[0] += -0.0198052332783761;
                    } else {
                      result[0] += -0.1258612596023134;
                    }
                  } else {
                    result[0] += 0.008591270895432148;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.397998809814454013) ) ) {
                  result[0] += 0.007852883449336257;
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.971427202224732333) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.102759599685669833) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.649621725082398349) ) ) {
                        result[0] += 0.0027132945869334743;
                      } else {
                        if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)279.5000000000000568) ) ) {
                          result[0] += 0.001781482847566398;
                        } else {
                          result[0] += -0.02651225634698098;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
                        result[0] += 0.016253453493424254;
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.986973047256470615) ) ) {
                          result[0] += 0.02060176600348905;
                        } else {
                          result[0] += -0.0819303525662678;
                        }
                      }
                    }
                  } else {
                    result[0] += -0.019889609586478268;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                  result[0] += -0.01988255463824276;
                } else {
                  result[0] += -0.07309464641795659;
                }
              } else {
                result[0] += -0.05833379822721932;
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.36986422538757413) ) ) {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.322819471359253818) ) ) {
            if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)326.5000000000000568) ) ) {
              result[0] += 0.0006872082697106014;
            } else {
              result[0] += 0.09329052768935468;
            }
          } else {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.06596770949593704;
            } else {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)302.5000000000000568) ) ) {
                result[0] += -0.031282238738766115;
              } else {
                result[0] += 0.013951805985651988;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.924581527709961826) ) ) {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)298.5000000000000568) ) ) {
                result[0] += 0.028584236270243337;
              } else {
                result[0] += -0.009709113874491632;
              }
            } else {
              result[0] += 0.07018686394141897;
            }
          } else {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
              if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                result[0] += 0.009032248515585017;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.924581527709961826) ) ) {
                  result[0] += 0.018546011766474815;
                } else {
                  result[0] += -0.13251868100397665;
                }
              }
            } else {
              result[0] += -0.04371139822368659;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
          if ( UNLIKELY( !(data[58].missing != -1) || (data[58].fvalue <= (double)3.000000000000000444) ) ) {
            result[0] += 0.005088772980039301;
          } else {
            result[0] += -0.008697120273601214;
          }
        } else {
          result[0] += 0.005672535906202034;
        }
      }
    }
  }
  if ( LIKELY( !(data[58].missing != -1) || (data[58].fvalue <= (double)1.500000000000000222) ) ) {
    result[0] += -0.000591304889671115;
  } else {
    if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
      if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)186.5000000000000284) ) ) {
        if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)0.8958797454833985485) ) ) {
          result[0] += -0.028722008307896514;
        } else {
          if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.497866153717041238) ) ) {
            if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.01416344987857485;
            } else {
              result[0] += -0.055489508215665076;
            }
          } else {
            result[0] += -0.0031956453868397816;
          }
        }
      } else {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.49584054946899592) ) ) {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
              if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)244.5000000000000284) ) ) {
                result[0] += -0.021868475857948506;
              } else {
                result[0] += 0.04879464859394613;
              }
            } else {
              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.700598716735840066) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.426736354827881748) ) ) {
                  if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)238.5000000000000284) ) ) {
                    if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.497866153717041238) ) ) {
                      result[0] += -0.01761250912865759;
                    } else {
                      result[0] += 0.04044404376058877;
                    }
                  } else {
                    result[0] += 0.026493864943595735;
                  }
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.617236852645874912) ) ) {
                    result[0] += -0.08075021127306627;
                  } else {
                    result[0] += 0.03463612007333738;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[56].missing != -1) || (data[56].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.637949228286744052) ) ) {
                    result[0] += -0.005558137482862763;
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
                      result[0] += 0.09315528886141228;
                    } else {
                      result[0] += 0.017619760258367624;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.744487762451173651) ) ) {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.531673669815064365) ) ) {
                        if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.349750161170959917) ) ) {
                          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.921060562133789951) ) ) {
                            result[0] += -0.10372724831937088;
                          } else {
                            result[0] += -0.04281926542493025;
                          }
                        } else {
                          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)312.5000000000000568) ) ) {
                            result[0] += 0.019918274497374204;
                          } else {
                            result[0] += -0.03977860620405704;
                          }
                        }
                      } else {
                        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.742733001708986151) ) ) {
                          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.397998809814454013) ) ) {
                            result[0] += -0.006263212505687563;
                          } else {
                            if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.740319490432739702) ) ) {
                              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)325.5000000000000568) ) ) {
                                result[0] += -0.008370847715603523;
                              } else {
                                result[0] += 0.030151274297170114;
                              }
                            } else {
                              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.861792564392090288) ) ) {
                                result[0] += 0.05917376138592926;
                              } else {
                                result[0] += 0.012290091408877539;
                              }
                            }
                          }
                        } else {
                          if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.03420138359069913) ) ) {
                            result[0] += -0.01831306413436774;
                          } else {
                            result[0] += -0.0815197407543335;
                          }
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.219419956207276279) ) ) {
                        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.855921268463135654) ) ) {
                          result[0] += 0.0013372475525768928;
                        } else {
                          result[0] += -0.03782154342634969;
                        }
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.029068946838379794) ) ) {
                          result[0] += 0.02745464799941337;
                        } else {
                          if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.855921268463135654) ) ) {
                            result[0] += 0.014269281624385018;
                          } else {
                            result[0] += -0.028741189981105742;
                          }
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.040618419647218573) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.590985536575318271) ) ) {
                        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.248013019561768466) ) ) {
                            result[0] += -0.007030261299605975;
                          } else {
                            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.53326439857482999) ) ) {
                              result[0] += -0.007089975975651708;
                            } else {
                              result[0] += -0.13342445473207185;
                            }
                          }
                        } else {
                          result[0] += 0.02679755464248537;
                        }
                      } else {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.384830474853516513) ) ) {
                          if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.238486170768738237) ) ) {
                            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.388278961181641513) ) ) {
                              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.363266706466675693) ) ) {
                                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.120439291000367099) ) ) {
                                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.285887241363526279) ) ) {
                                    result[0] += 0.024439726373605527;
                                  } else {
                                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.084203958511353427) ) ) {
                                      result[0] += -0.030672666722715278;
                                    } else {
                                      result[0] += 0.02874022156783187;
                                    }
                                  }
                                } else {
                                  result[0] += 0.021829113720423787;
                                }
                              } else {
                                result[0] += -0.07258697288668707;
                              }
                            } else {
                              if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                                result[0] += -0.024213412338172386;
                              } else {
                                result[0] += 0.06045360728469799;
                              }
                            }
                          } else {
                            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.029068946838379794) ) ) {
                              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.255632162094117099) ) ) {
                                result[0] += 0.04358198555176067;
                              } else {
                                result[0] += 0.00872553969821067;
                              }
                            } else {
                              result[0] += -0.006885898825914571;
                            }
                          }
                        } else {
                          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.450390577316285068) ) ) {
                              result[0] += -0.018723388574738986;
                            } else {
                              if ( UNLIKELY( !(data[58].missing != -1) || (data[58].fvalue <= (double)3.000000000000000444) ) ) {
                                result[0] += 0.044677821350875745;
                              } else {
                                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.040716171264650214) ) ) {
                                  result[0] += -0.06318953383398289;
                                } else {
                                  result[0] += 0.033797263438784045;
                                }
                              }
                            }
                          } else {
                            result[0] += -0.07682837578921263;
                          }
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.029068946838379794) ) ) {
                        if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.650573849678039995) ) ) {
                          result[0] += -0.020261476391140742;
                        } else {
                          result[0] += 0.0019279143025522272;
                        }
                      } else {
                        result[0] += 0.004138253649896763;
                      }
                    }
                  }
                }
              }
            }
          } else {
            result[0] += 0.01608249500319514;
          }
        } else {
          if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)233.5000000000000284) ) ) {
            result[0] += 0.005911188306126895;
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.558241367340089667) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.637949228286744052) ) ) {
                if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                  result[0] += -0.03746223535300085;
                } else {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.0031062650026652087;
                  } else {
                    if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)302.5000000000000568) ) ) {
                      result[0] += -0.006843701668963415;
                    } else {
                      result[0] += -0.044443362941003;
                    }
                  }
                }
              } else {
                result[0] += 0.0010454398687472107;
              }
            } else {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += -0.021687086913565558;
              } else {
                result[0] += -0.0548083608207196;
              }
            }
          }
        }
      }
    } else {
      result[0] += 0.003627776720401438;
    }
  }
  if ( LIKELY( !(data[58].missing != -1) || (data[58].fvalue <= (double)1.500000000000000222) ) ) {
    result[0] += -0.0005900689216871942;
  } else {
    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
      if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)186.5000000000000284) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.42478513717651456) ) ) {
            result[0] += -0.007558840059587974;
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.04496813681165662;
            } else {
              result[0] += 0.010317285311469141;
            }
          }
        } else {
          if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.790116786956788886) ) ) {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.173939466476441318) ) ) {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.851041555404663974) ) ) {
                    result[0] += -0.016878917700538874;
                  } else {
                    result[0] += -0.08698138350374796;
                  }
                } else {
                  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += -0.04571598150752973;
                  } else {
                    result[0] += 0.006722960126656262;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.009206703142291748;
                } else {
                  result[0] += 0.006464084761310256;
                }
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.855006217956543857) ) ) {
                if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                  result[0] += 0.0077301550778471236;
                } else {
                  result[0] += -0.02578932100386855;
                }
              } else {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += -0.04347534182893264;
                      } else {
                        result[0] += 0.033819479965667996;
                      }
                    } else {
                      result[0] += 0.027438500406720353;
                    }
                  } else {
                    result[0] += -0.003263179604468744;
                  }
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.192109584808350498) ) ) {
                    result[0] += -0.023098525328737362;
                  } else {
                    result[0] += 0.025128742562701357;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.40000796318054288) ) ) {
              result[0] += 0.005450495498055854;
            } else {
              result[0] += -0.047471883305120616;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.49584054946899592) ) ) {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)244.5000000000000284) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.285887241363526279) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.625595092773438388) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.996674776077271396) ) ) {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.534971714019776279) ) ) {
                      result[0] += -0.005036354506893424;
                    } else {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.743881702423096591) ) ) {
                        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
                          result[0] += -0.07191455242677627;
                        } else {
                          if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                            result[0] += -0.05182886098739449;
                          } else {
                            result[0] += 0.03904438224755623;
                          }
                        }
                      } else {
                        result[0] += -0.012653814922750362;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.901921629905701128) ) ) {
                      result[0] += -0.07725191493490505;
                    } else {
                      result[0] += 0.004759011876566744;
                    }
                  }
                } else {
                  result[0] += 0.004869451296086651;
                }
              } else {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                  if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)192.5000000000000284) ) ) {
                    result[0] += 0.02072322762413779;
                  } else {
                    result[0] += -0.021857724534705743;
                  }
                } else {
                  result[0] += -0.05641168061894443;
                }
              }
            } else {
              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.700598716735840066) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.426736354827881748) ) ) {
                  result[0] += 0.024300817202972492;
                } else {
                  result[0] += -0.07382158183453945;
                }
              } else {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                  result[0] += 0.04810871580907111;
                } else {
                  if ( UNLIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += 0.014732939353214812;
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.553655147552491123) ) ) {
                      if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += -0.008716420428908847;
                      } else {
                        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.481121778488159624) ) ) {
                          result[0] += -0.00505517842220297;
                        } else {
                          result[0] += -0.16293640675466387;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)6.809154510498047763) ) ) {
                        result[0] += 0.002594761990072874;
                      } else {
                        result[0] += 0.05310073768391798;
                      }
                    }
                  }
                }
              }
            }
          } else {
            result[0] += 0.015275834497029392;
          }
        } else {
          if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)233.5000000000000284) ) ) {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
              result[0] += 0.004423549275166458;
            } else {
              if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.059697883129552244;
              } else {
                result[0] += -0.04859942317937864;
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.558241367340089667) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.637949228286744052) ) ) {
                if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                  result[0] += -0.03497728421541981;
                } else {
                  if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)314.5000000000000568) ) ) {
                    result[0] += 0.002468634905520644;
                  } else {
                    result[0] += -0.02175895020908282;
                  }
                }
              } else {
                result[0] += 0.0014092909268016754;
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                result[0] += 0.003693455460126811;
              } else {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.851041555404663974) ) ) {
                    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.02562677645183426;
                    } else {
                      result[0] += -0.07787548304229157;
                    }
                  } else {
                    result[0] += -0.0035476202571395987;
                  }
                } else {
                  result[0] += -0.06323632969260234;
                }
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.97887301445007413) ) ) {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
          if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.322819471359253818) ) ) {
              if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += -0.002728546021443494;
              } else {
                result[0] += 0.07248076498116443;
              }
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                result[0] += -0.03328904105269843;
              } else {
                if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)326.5000000000000568) ) ) {
                  if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)168.5000000000000284) ) ) {
                    result[0] += -0.044709989350348825;
                  } else {
                    result[0] += 0.008123576159410544;
                  }
                } else {
                  result[0] += -0.08438953470375457;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
              result[0] += 0.18063718436856227;
            } else {
              result[0] += 0.01573089174130318;
            }
          }
        } else {
          result[0] += 0.017285400450149833;
        }
      } else {
        result[0] += 0.004002143911046683;
      }
    }
  }
  if ( LIKELY( !(data[58].missing != -1) || (data[58].fvalue <= (double)1.500000000000000222) ) ) {
    if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.363266706466675693) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.835998296737671787) ) ) {
          result[0] += 0.06838434400533004;
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.029068946838379794) ) ) {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += 0.026767325538526895;
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.590985536575318271) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.623839378356934482) ) ) {
                  if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.012675821781158891) ) ) {
                    result[0] += -0.015272878681758842;
                  } else {
                    if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                      result[0] += -0.038898347684627216;
                    } else {
                      result[0] += -0.11405803315593246;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.373361587524414951) ) ) {
                    result[0] += 0.017329809973146003;
                  } else {
                    result[0] += -0.05793871713222938;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.843275547027588779) ) ) {
                  result[0] += -0.03249601218488265;
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.715336322784424716) ) ) {
                    result[0] += 0.065045056106667;
                  } else {
                    result[0] += 0.0014400129732317405;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.637949228286744052) ) ) {
              if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.644374847412110263) ) ) {
                  result[0] += 0.02879717115354724;
                } else {
                  result[0] += 0.11579964541823827;
                }
              } else {
                result[0] += -0.04604173109533178;
              }
            } else {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.038351853687818044;
              } else {
                if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += 0.02985777790481446;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.58491539955139249) ) ) {
                    result[0] += 0.0841909012210153;
                  } else {
                    result[0] += -0.025217797922995333;
                  }
                }
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.61675357818603693) ) ) {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.262283086776734287) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.040716171264650214) ) ) {
                      result[0] += -0.050129427055922576;
                    } else {
                      result[0] += 0.025580354672414842;
                    }
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.69067406654357999) ) ) {
                      result[0] += -0.10142879713559834;
                    } else {
                      result[0] += -0.013270390883870643;
                    }
                  }
                } else {
                  result[0] += 0.05392182543466818;
                }
              } else {
                if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.617236852645874912) ) ) {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.597323656082154208) ) ) {
                      result[0] += -0.001935931514777253;
                    } else {
                      result[0] += -0.041283941596032825;
                    }
                  } else {
                    result[0] += -0.07583048934281493;
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.632926940917970526) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                      result[0] += -0.009772506599062099;
                    } else {
                      result[0] += -0.09183726599313563;
                    }
                  } else {
                    result[0] += -0.07260297634912134;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.921100616455079013) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.24049568176269709) ) ) {
                  if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += 0.048134836212017036;
                  } else {
                    result[0] += -0.0960950597602061;
                  }
                } else {
                  result[0] += 0.0077486642103125215;
                }
              } else {
                if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += -0.012546494926773365;
                } else {
                  result[0] += -0.07517517046337971;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.142630577087403232) ) ) {
                result[0] += -0.057957316709721624;
              } else {
                result[0] += -0.007514398838639057;
              }
            } else {
              result[0] += -0.07039894201162859;
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.190353393554689276) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.216319084167481357) ) ) {
              if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.049816085961558315;
              } else {
                result[0] += -0.02204974624963021;
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.921100616455079013) ) ) {
                result[0] += -0.08056945595531387;
              } else {
                result[0] += -0.01057266709028553;
              }
            }
          } else {
            result[0] += -0.05937045891738664;
          }
        }
      }
    } else {
      result[0] += -0.0008215099494874893;
    }
  } else {
    if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)6144.000000000000909) ) ) {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)5.500000000000000888) ) ) {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
          result[0] += 0.00022304440086334;
        } else {
          if ( UNLIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)6.000000000000000888) ) ) {
            result[0] += -0.0038929669710324605;
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.97887301445007413) ) ) {
              if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.553655147552491123) ) ) {
                  result[0] += 0.05194137272004351;
                } else {
                  result[0] += -0.009876006836003086;
                }
              } else {
                result[0] += 0.035626605162391654;
              }
            } else {
              result[0] += 0.0045939653591686395;
            }
          }
        }
      } else {
        result[0] += -0.06841324418325541;
      }
    } else {
      if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
        if ( LIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)48.00000000000000711) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.623839378356934482) ) ) {
            result[0] += 0.024728108108044974;
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.120439291000367099) ) ) {
              result[0] += 0.01449797470817004;
            } else {
              if ( LIKELY( !(data[56].missing != -1) || (data[56].fvalue <= (double)6.000000000000000888) ) ) {
                if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += -0.11346126187593407;
                } else {
                  result[0] += -0.01004556612432521;
                }
              } else {
                result[0] += -0.13284808488168115;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.040716171264650214) ) ) {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += 0.03677149636658554;
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.531007289886475498) ) ) {
                result[0] += -0.06220405992967072;
              } else {
                result[0] += 0.08623852027669693;
              }
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.342454433441162998) ) ) {
              result[0] += -0.032479036908804826;
            } else {
              result[0] += 0.07092759600439036;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.623839378356934482) ) ) {
          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
            result[0] += -0.028766281137520805;
          } else {
            result[0] += -0.10704099986303311;
          }
        } else {
          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
            result[0] += 0.05354743573436983;
          } else {
            result[0] += -0.03797059786130314;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[58].missing != -1) || (data[58].fvalue <= (double)1.500000000000000222) ) ) {
    if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.363266706466675693) ) ) {
        if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)91.50000000000001421) ) ) {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.552201986312867099) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.36986422538757413) ) ) {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.623839378356934482) ) ) {
                  if ( UNLIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                      result[0] += -0.013689138004142494;
                    } else {
                      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += -0.09246103617829363;
                      } else {
                        result[0] += -0.038163676595314194;
                      }
                    }
                  } else {
                    result[0] += 0.007500719914535827;
                  }
                } else {
                  if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.487163543701172763) ) ) {
                    result[0] += 0.01073899450399748;
                  } else {
                    result[0] += -0.0845695364297403;
                  }
                }
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.637949228286744052) ) ) {
                  result[0] += 0.02408895247165189;
                } else {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                    result[0] += -0.02879111422670065;
                  } else {
                    result[0] += 0.03557485669309204;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.572941064834595615) ) ) {
                  result[0] += 0.1345880586064094;
                } else {
                  result[0] += 0.0322024505157388;
                }
              } else {
                if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.020127415657043901) ) ) {
                  result[0] += 0.01845578525475547;
                } else {
                  result[0] += -0.09546022102629181;
                }
              }
            }
          } else {
            result[0] += 0.06176131371818229;
          }
        } else {
          if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.987706661224366123) ) ) {
            result[0] += 0.05167306463063753;
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
              result[0] += 0.023468479925509603;
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                result[0] += -0.005532816549968479;
              } else {
                result[0] += -0.08532711398141646;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.37109279632568537) ) ) {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.918272972106934482) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.262283086776734287) ) ) {
                    result[0] += 0.021843781595192308;
                  } else {
                    result[0] += -0.016407793166012326;
                  }
                } else {
                  result[0] += 0.049266789748543004;
                }
              } else {
                if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                  if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.553655147552491123) ) ) {
                      result[0] += 0.09478803504685776;
                    } else {
                      result[0] += -0.027338805626653164;
                    }
                  } else {
                    if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)91.50000000000001421) ) ) {
                      result[0] += -0.004679526940497564;
                    } else {
                      result[0] += 0.0368815941794979;
                    }
                  }
                } else {
                  result[0] += -0.047274870169216276;
                }
              }
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.921100616455079013) ) ) {
                if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.238486170768738237) ) ) {
                  result[0] += 0.0077818853099531115;
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.26837396621704279) ) ) {
                    result[0] += 0.05128867889954877;
                  } else {
                    result[0] += 0.01421278070664904;
                  }
                }
              } else {
                if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += -0.01085101091536891;
                } else {
                  result[0] += -0.0728304245079088;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.141444921493531162) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.802696108818054643) ) ) {
                result[0] += -0.05178909222328654;
              } else {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)87.50000000000001421) ) ) {
                  result[0] += 0.018720709118766023;
                } else {
                  result[0] += -0.017712848127427887;
                }
              }
            } else {
              result[0] += -0.05930829596362524;
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.190353393554689276) ) ) {
            result[0] += -0.014959777927201543;
          } else {
            result[0] += -0.05515030298733567;
          }
        }
      }
    } else {
      result[0] += -0.0008230138099201275;
    }
  } else {
    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
      if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)100.5000000000000142) ) ) {
        result[0] += -0.056673022997599964;
      } else {
        result[0] += 0.01640529200686616;
      }
    } else {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
        if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)186.5000000000000284) ) ) {
          if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.863673448562622958) ) ) {
            result[0] += -0.003632933621392063;
          } else {
            if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.023306644141435844;
              } else {
                result[0] += -0.008219273895774301;
              }
            } else {
              result[0] += -0.06539167450807891;
            }
          }
        } else {
          if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.49584054946899592) ) ) {
              if ( UNLIKELY( !(data[56].missing != -1) || (data[56].fvalue <= (double)3.000000000000000444) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.029068946838379794) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.43267917633056818) ) ) {
                    result[0] += -0.004613582225766938;
                  } else {
                    result[0] += -0.10501667544877076;
                  }
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.426736354827881748) ) ) {
                    result[0] += 0.017775171299376374;
                  } else {
                    result[0] += -0.033141690839556655;
                  }
                }
              } else {
                result[0] += 0.0010097160607880403;
              }
            } else {
              result[0] += 0.014140333405975048;
            }
          } else {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.558241367340089667) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.617236852645874912) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.055496215820313388) ) ) {
                    result[0] += 0.0016418196890154177;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.649621725082398349) ) ) {
                      result[0] += 0.0005213371056113287;
                    } else {
                      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.289595603942871982) ) ) {
                        result[0] += -0.01718772499066974;
                      } else {
                        result[0] += -0.04199702969320165;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.0012168205389459388;
                    } else {
                      result[0] += 0.03729910635170883;
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.590985536575318271) ) ) {
                      result[0] += 0.01319777868492785;
                    } else {
                      result[0] += -0.034507228045078296;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.918272972106934482) ) ) {
                  result[0] += 0.0038706597467629473;
                } else {
                  result[0] += -0.029888570876721046;
                }
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                result[0] += 0.0007418187858165591;
              } else {
                result[0] += 0.01195042467774953;
              }
            }
          }
        }
      } else {
        result[0] += 0.0031654285006298233;
      }
    }
  }
  if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)6.000000000000000888) ) ) {
    result[0] += 0.00018639386953944555;
  } else {
    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
      if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)0.8958797454833985485) ) ) {
        if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)136.5000000000000284) ) ) {
          result[0] += -0.052140280047178855;
        } else {
          result[0] += -0.01325186537669574;
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
            result[0] += -0.069943702332252;
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.028861761093140537) ) ) {
              if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)262.5000000000000568) ) ) {
                  result[0] += -0.014442919843201191;
                } else {
                  result[0] += 0.020044474092712233;
                }
              } else {
                if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.24492526054382413) ) ) {
                    result[0] += 0.14662569837296832;
                  } else {
                    result[0] += 0.007960890345368394;
                  }
                } else {
                  if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.238486170768738237) ) ) {
                    if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)176.5000000000000284) ) ) {
                      if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += 0.022476905143951988;
                      } else {
                        result[0] += -0.010531311379908256;
                      }
                    } else {
                      result[0] += -5.150288830558537e-06;
                    }
                  } else {
                    result[0] += 0.03559789328942993;
                  }
                }
              }
            } else {
              result[0] += -0.034105496199784056;
            }
          }
        } else {
          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)273.5000000000000568) ) ) {
            if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.662244915962219682) ) ) {
                  result[0] += 0.012690996305280276;
                } else {
                  result[0] += -0.0659447590752724;
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.662244915962219682) ) ) {
                  result[0] += 0.09885904192071215;
                } else {
                  result[0] += 0.006206027220532511;
                }
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.247576236724854404) ) ) {
                if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.071567356586456743) ) ) {
                    if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
                      result[0] += 0.007889901051235352;
                    } else {
                      result[0] += -0.06559283694257959;
                    }
                  } else {
                    result[0] += 0.031814327723021694;
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.439304351806642401) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.131699204444885698) ) ) {
                      result[0] += 0.24221168266215073;
                    } else {
                      result[0] += 0.013485427848361271;
                    }
                  } else {
                    result[0] += -0.03816062405315543;
                  }
                }
              } else {
                if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)136.5000000000000284) ) ) {
                  if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
                      if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)48.00000000000000711) ) ) {
                        result[0] += 0.01622386479909111;
                      } else {
                        result[0] += -0.02768779829922191;
                      }
                    } else {
                      result[0] += -0.042990337965849675;
                    }
                  } else {
                    result[0] += 0.005933317252071473;
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.69067406654357999) ) ) {
                    result[0] += -0.013312180170846791;
                  } else {
                    result[0] += 0.011704427819335327;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.025192260742188388) ) ) {
                result[0] += -0.026373737941909326;
              } else {
                result[0] += 0.008356709163376755;
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                result[0] += -0.012089735806875583;
              } else {
                result[0] += -0.0492983000716395;
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
        if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)7.27828097343444913) ) ) {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.835998296737671787) ) ) {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)262.5000000000000568) ) ) {
                  if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                    result[0] += -0.03689152862292086;
                  } else {
                    result[0] += -0.0021728009803440194;
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.556798219680787021) ) ) {
                    result[0] += 0.09265658164807018;
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.497866153717041238) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.040716171264650214) ) ) {
                        result[0] += -0.010125224546310957;
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.742733001708986151) ) ) {
                          result[0] += 0.11219307356524502;
                        } else {
                          result[0] += 0.010371193613700017;
                        }
                      }
                    } else {
                      result[0] += -0.030038587078529994;
                    }
                  }
                }
              } else {
                result[0] += 0.020604567488447856;
              }
            } else {
              if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.467917680740357333) ) ) {
                  result[0] += -0.02359262444290793;
                } else {
                  result[0] += -0.056563220728871266;
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.989220380783081943) ) ) {
                  result[0] += -0.043371072357734616;
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.777633190155030185) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.53326439857482999) ) ) {
                      result[0] += -0.0013102780883544104;
                    } else {
                      result[0] += -0.06719977243923078;
                    }
                  } else {
                    result[0] += 0.0244117516572025;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.951942920684815341) ) ) {
              result[0] += 0.013127892026432247;
            } else {
              result[0] += -0.022551327605156605;
            }
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.835998296737671787) ) ) {
            if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.242453336715698464) ) ) {
              result[0] += -0.01957188679487193;
            } else {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.09668002876812443;
              } else {
                result[0] += -0.012694635585665748;
              }
            }
          } else {
            result[0] += -0.07348349004550978;
          }
        }
      } else {
        if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.92430353164673029) ) ) {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)121.5000000000000142) ) ) {
              result[0] += -0.08705884498278266;
            } else {
              if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5.695412874221802646) ) ) {
                if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.337269306182862216) ) ) {
                  result[0] += 0.011046436940004564;
                } else {
                  result[0] += -0.04341712357909623;
                }
              } else {
                result[0] += 0.14131896053148765;
              }
            }
          } else {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
                if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.01634240150451749) ) ) {
                  result[0] += 0.006202412891580231;
                } else {
                  result[0] += 0.07487624126557794;
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)14.98329114913940607) ) ) {
                  result[0] += 0.03794605449102169;
                } else {
                  result[0] += -0.02751337201931068;
                }
              }
            } else {
              result[0] += -0.05857300261610528;
            }
          }
        } else {
          result[0] += -0.007666830830460602;
        }
      }
    }
  }
  if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
    result[0] += -0.0004388368447981268;
  } else {
    if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.102609157562256748) ) ) {
      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.51693725585937678) ) ) {
        if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.507949829101563388) ) ) {
            if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.802696108818054643) ) ) {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.524927973747253862) ) ) {
                  result[0] += 0.01803284995645988;
                } else {
                  result[0] += -0.029251376898324168;
                }
              } else {
                result[0] += -2.9800840716305383e-05;
              }
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.761470437049866167) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.216319084167481357) ) ) {
                  if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.349750161170959917) ) ) {
                    if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.012675821781158891) ) ) {
                      result[0] += -0.011931948011347162;
                    } else {
                      result[0] += 0.04952111691200506;
                    }
                  } else {
                    result[0] += -0.0020361206961702328;
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.932935476303101474) ) ) {
                    result[0] += 0.020851604802298848;
                  } else {
                    result[0] += -0.03005821589921383;
                  }
                }
              } else {
                result[0] += 0.10489219111889017;
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.972535848617554599) ) ) {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.334978580474854404) ) ) {
                  result[0] += -0.07025732748092454;
                } else {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.07507595340363041;
                  } else {
                    result[0] += 0.028083716432882485;
                  }
                }
              } else {
                result[0] += 0.05708219676652041;
              }
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
                if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)294.5000000000000568) ) ) {
                    result[0] += -0.03971215148548338;
                  } else {
                    result[0] += 0.016088001306799462;
                  }
                } else {
                  result[0] += 0.025838677196729488;
                }
              } else {
                result[0] += -0.08364711178099664;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
            if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.085941076278687412) ) ) {
              result[0] += 0.004284630855139424;
            } else {
              result[0] += -0.09686059930504398;
            }
          } else {
            if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
              if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.835998296737671787) ) ) {
                  result[0] += -0.10861297068411589;
                } else {
                  result[0] += -0.013168721902415909;
                }
              } else {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                  result[0] += 0.013312056070806612;
                } else {
                  result[0] += -0.02313354207074812;
                }
              }
            } else {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.921924352645874468) ) ) {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
                    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.028429772188202397;
                    } else {
                      result[0] += -0.03433435233740516;
                    }
                  } else {
                    if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += -0.08130120438025315;
                    } else {
                      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.276966691017151323) ) ) {
                        result[0] += -0.04954613563125281;
                      } else {
                        result[0] += -0.011944691101320258;
                      }
                    }
                  }
                } else {
                  result[0] += -0.015459921034433078;
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.835998296737671787) ) ) {
                  result[0] += -0.2245118499271039;
                } else {
                  result[0] += -0.0016484231608236326;
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
            result[0] += 0.0024043161350813347;
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.262283086776734287) ) ) {
              result[0] += -0.022847445159020097;
            } else {
              if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                result[0] += 0.018458044511852283;
              } else {
                result[0] += 0.040071144655845436;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)197.5000000000000284) ) ) {
            if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.921060562133789951) ) ) {
                result[0] += -0.03375943633327324;
              } else {
                result[0] += 0.009439460213614113;
              }
            } else {
              if ( UNLIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += 0.032380928648030764;
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.53326439857482999) ) ) {
                  result[0] += -0.05808533923210151;
                } else {
                  if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.03248383410986114;
                    } else {
                      result[0] += -0.021141990529320073;
                    }
                  } else {
                    result[0] += -0.09508633972119483;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += -0.011734414687792198;
            } else {
              result[0] += -0.05631560752747314;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.662244915962219682) ) ) {
        result[0] += -0.017861778755920772;
      } else {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.040618419647218573) ) ) {
            result[0] += -0.018345988631294288;
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.553655147552491123) ) ) {
              if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.938033580780031073) ) ) {
                  result[0] += -0.01656489450276624;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.9353518486022967) ) ) {
                    result[0] += 0.15824312000660887;
                  } else {
                    result[0] += 0.048596888574046676;
                  }
                }
              } else {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.029068946838379794) ) ) {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.013177497498491751;
                  } else {
                    if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.006005879505032479;
                    } else {
                      result[0] += -0.06803947901824678;
                    }
                  }
                } else {
                  result[0] += -0.06366051965246221;
                }
              }
            } else {
              result[0] += 0.01610862022273372;
            }
          }
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.537586927413940874) ) ) {
            if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += 0.013223520755817321;
            } else {
              if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.023225734839246916;
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                      result[0] += -0.030828106268544653;
                    } else {
                      result[0] += 0.026315570074319894;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                    result[0] += -0.018764522539623435;
                  } else {
                    result[0] += -0.1013642831211533;
                  }
                }
              } else {
                result[0] += -0.08050054371786423;
              }
            }
          } else {
            result[0] += 0.02115723180078133;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
    result[0] += -0.00045244590553554666;
  } else {
    if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.102609157562256748) ) ) {
      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.51693725585937678) ) ) {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.539549827575684482) ) ) {
          if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.43450713157653853) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.102759599685669833) ) ) {
                  result[0] += -0.0015497314988694307;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.679712533950806552) ) ) {
                    if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += 0.007104262635780942;
                    } else {
                      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += -0.09458493217714035;
                      } else {
                        result[0] += -0.03857290297687527;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.01845818704365552;
                    } else {
                      if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.569433569908142534) ) ) {
                          result[0] += -0.1596672638473308;
                        } else {
                          result[0] += -0.03239560518145876;
                        }
                      } else {
                        result[0] += -0.009550569338882296;
                      }
                    }
                  }
                }
              } else {
                result[0] += -0.08582682758295204;
              }
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.102759599685669833) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.764287948608400214) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.464211463928224433) ) ) {
                    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
                        result[0] += -0.004211173565774522;
                      } else {
                        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                          result[0] += 0.0026153779898713697;
                        } else {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.120439291000367099) ) ) {
                            result[0] += 0.00886159501922537;
                          } else {
                            result[0] += 0.06959441692593482;
                          }
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.349750161170959917) ) ) {
                        if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.132848501205445224) ) ) {
                          result[0] += 0.017835716943265897;
                        } else {
                          result[0] += -0.025278559479407867;
                        }
                      } else {
                        result[0] += 0.03607737296638951;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.285887241363526279) ) ) {
                      result[0] += 0.027384530705862983;
                    } else {
                      result[0] += -0.04402990514823543;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.53326439857482999) ) ) {
                    result[0] += -0.03721951313433333;
                  } else {
                    if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                        result[0] += -0.018482852371079456;
                      } else {
                        result[0] += 0.01721251809440244;
                      }
                    } else {
                      if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.53326439857482999) ) ) {
                          result[0] += 0.10751515727482677;
                        } else {
                          result[0] += 0.0008517712342348423;
                        }
                      } else {
                        result[0] += -0.03338238237715441;
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.673553824424744096) ) ) {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                          result[0] += 0.04119661846769813;
                        } else {
                          result[0] += -0.0979123461484235;
                        }
                      } else {
                        result[0] += -0.1491237247931694;
                      }
                    } else {
                      if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.602003335952759233) ) ) {
                        result[0] += -0.028088518041983382;
                      } else {
                        result[0] += 0.02354494239328877;
                      }
                    }
                  } else {
                    result[0] += 0.04729261830561005;
                  }
                } else {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
                    result[0] += 0.03619653454560407;
                  } else {
                    result[0] += -0.08329491955710143;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.43450713157653853) ) ) {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                result[0] += 0.014301846634542831;
              } else {
                result[0] += -0.016901653738958827;
              }
            } else {
              result[0] += 0.08849355634123421;
            }
          }
        } else {
          if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.138333082199097124) ) ) {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.012675821781158891) ) ) {
                result[0] += -0.028616936615100008;
              } else {
                result[0] += 0.01024856303698167;
              }
            } else {
              result[0] += 0.024199715830824888;
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)8.285748958587648261) ) ) {
              result[0] += -0.004174436908571214;
            } else {
              result[0] += 0.04790476719120182;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += 0.014675649384762955;
        } else {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += -0.010360256697845245;
            } else {
              result[0] += -0.05128407336989865;
            }
          } else {
            if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
              result[0] += -0.01716904978915917;
            } else {
              if ( UNLIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += 0.030746749507712653;
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
                  result[0] += -0.022607909086346852;
                } else {
                  result[0] += 0.03592447695137176;
                }
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.553655147552491123) ) ) {
        if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.142630577087403232) ) ) {
            result[0] += -0.05483225253152424;
          } else {
            result[0] += 0.06147173899842717;
          }
        } else {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.029068946838379794) ) ) {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.004297752152546722;
            } else {
              result[0] += -0.029482962900901955;
            }
          } else {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += 0.04417540322449123;
            } else {
              result[0] += -0.08434980246600837;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.126503944396974433) ) ) {
            if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)7.223295450210572177) ) ) {
              result[0] += 0.014106573851703783;
            } else {
              result[0] += -0.08029638236221937;
            }
          } else {
            result[0] += 0.012341686303327793;
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.873467922210695136) ) ) {
            result[0] += 0.030919913065178196;
          } else {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
              result[0] += 0.08422588688884136;
            } else {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.53326439857482999) ) ) {
                  if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                    result[0] += 0.06431665611171831;
                  } else {
                    result[0] += -0.08320942568801659;
                  }
                } else {
                  result[0] += -0.01746268871787771;
                }
              } else {
                result[0] += -0.05527932394286561;
              }
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
    result[0] += -0.0004551586140519998;
  } else {
    if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.102609157562256748) ) ) {
      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.51693725585937678) ) ) {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.388278961181641513) ) ) {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.43450713157653853) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.587308406829834873) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.141444921493531162) ) ) {
                    if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.088880300521851474) ) ) {
                        result[0] += 0.008356628275133195;
                      } else {
                        result[0] += -0.004886980242567344;
                      }
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.835998296737671787) ) ) {
                        result[0] += 0.020985330782436903;
                      } else {
                        result[0] += -0.03611031838557612;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)299.5000000000000568) ) ) {
                      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.102759599685669833) ) ) {
                        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                          result[0] += -0.020570097482131185;
                        } else {
                          result[0] += -0.06338021748052292;
                        }
                      } else {
                        result[0] += -0.06176266076349061;
                      }
                    } else {
                      result[0] += 0.012489635680403174;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.088880300521851474) ) ) {
                    if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.420236110687257636) ) ) {
                        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.744487762451173651) ) ) {
                          if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                            result[0] += 0.06647055714856785;
                          } else {
                            result[0] += 0.0006020503303967635;
                          }
                        } else {
                          result[0] += 0.04951116038367018;
                        }
                      } else {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.802696108818054643) ) ) {
                          result[0] += -0.06298659611724097;
                        } else {
                          result[0] += -0.0021339110943273936;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.011523246765138495) ) ) {
                        if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)148.5000000000000284) ) ) {
                          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            result[0] += -0.01029389546917299;
                          } else {
                            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
                              result[0] += 0.10623938924935179;
                            } else {
                              result[0] += -0.06278121543656853;
                            }
                          }
                        } else {
                          if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.970085620880127397) ) ) {
                            result[0] += -0.015599693706463306;
                          } else {
                            if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                              result[0] += -0.01607098409620188;
                            } else {
                              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.190353393554689276) ) ) {
                                result[0] += 0.02548684485299591;
                              } else {
                                if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                                  if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                                    result[0] += -0.04735605556378263;
                                  } else {
                                    if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.481121778488159624) ) ) {
                                      result[0] += 0.06835745911643891;
                                    } else {
                                      result[0] += -0.030007670061343802;
                                    }
                                  }
                                } else {
                                  result[0] += 0.010676957763854895;
                                }
                              }
                            }
                          }
                        }
                      } else {
                        if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += -0.00732842425581216;
                        } else {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.512576580047609198) ) ) {
                            if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.700598716735840066) ) ) {
                              if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                                result[0] += 0.0231840966704904;
                              } else {
                                result[0] += -0.0169358045127198;
                              }
                            } else {
                              if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                                result[0] += -0.14871435426443916;
                              } else {
                                result[0] += -0.0005897207607171405;
                              }
                            }
                          } else {
                            result[0] += 0.009502188263024085;
                          }
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.349750161170959917) ) ) {
                        if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)190.5000000000000284) ) ) {
                          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.102759599685669833) ) ) {
                            result[0] += 0.041385583256744525;
                          } else {
                            result[0] += -0.053661341556115294;
                          }
                        } else {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.773543357849121982) ) ) {
                            result[0] += -0.06508001546572041;
                          } else {
                            result[0] += 0.0184386907525972;
                          }
                        }
                      } else {
                        if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.384246587753296343) ) ) {
                          result[0] += -0.038187545666491546;
                        } else {
                          result[0] += 0.025661284789301617;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.637949228286744052) ) ) {
                        result[0] += -0.001712855657843143;
                      } else {
                        if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.028861761093140537) ) ) {
                            result[0] += 0.002178801261719899;
                          } else {
                            result[0] += -0.06532010955832014;
                          }
                        } else {
                          result[0] += 0.011545745271769726;
                        }
                      }
                    }
                  }
                }
              } else {
                result[0] += -0.07965881226308497;
              }
            } else {
              result[0] += 0.019308755570988694;
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.102759599685669833) ) ) {
              if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.764287948608400214) ) ) {
                  if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)267.5000000000000568) ) ) {
                    result[0] += 0.006133616268222511;
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.742733001708986151) ) ) {
                      result[0] += -0.008451189674368248;
                    } else {
                      result[0] += 0.05376778834439625;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.53326439857482999) ) ) {
                    result[0] += -0.03663894037843903;
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.835998296737671787) ) ) {
                      result[0] += 0.03944731786101846;
                    } else {
                      if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)111.5000000000000142) ) ) {
                        result[0] += 0.01360432920534045;
                      } else {
                        if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                          result[0] += -0.006829080883572387;
                        } else {
                          result[0] += -0.031195100767836825;
                        }
                      }
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.013839965039998453;
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.623839378356934482) ) ) {
                    result[0] += 0.03928956658643316;
                  } else {
                    result[0] += 0.18358784956803134;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += 0.013947457971083028;
              } else {
                if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)278.5000000000000568) ) ) {
                  result[0] += 0.10316320972182345;
                } else {
                  result[0] += -0.00596989092298791;
                }
              }
            }
          }
        } else {
          result[0] += 0.005631957643398846;
        }
      } else {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
          result[0] += 0.010723997246457902;
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.921060562133789951) ) ) {
            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += -0.016338414463696462;
            } else {
              result[0] += -0.06145054091690998;
            }
          } else {
            result[0] += 0.012832634840567438;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.662244915962219682) ) ) {
        result[0] += -0.014969660033278959;
      } else {
        result[0] += 0.00869483503984967;
      }
    }
  }
  if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
    if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.795426130294800249) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)2.012675821781158891) ) ) {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
            result[0] += 0.02592141951770694;
          } else {
            result[0] += 0.12115998383363263;
          }
        } else {
          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.684611082077027255) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.241523027420044833) ) ) {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                  if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.397998809814454013) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.835998296737671787) ) ) {
                        result[0] += 0.006874424708452118;
                      } else {
                        result[0] += -0.05945321878771803;
                      }
                    } else {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.289602279663086826) ) ) {
                        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                          result[0] += -0.005398204342462539;
                        } else {
                          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.553655147552491123) ) ) {
                            result[0] += -0.12624453631386123;
                          } else {
                            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.58491539955139249) ) ) {
                              result[0] += -0.0891469350574613;
                            } else {
                              result[0] += -0.010649034693780195;
                            }
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += 0.018963988323273118;
                        } else {
                          result[0] += -0.01655811364606471;
                        }
                      }
                    }
                  } else {
                    result[0] += 0.020375889515973578;
                  }
                } else {
                  result[0] += 0.028965064264302832;
                }
              } else {
                if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.02499706381132326;
                } else {
                  if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)3.901921629905701128) ) ) {
                    result[0] += 0.02790647726609056;
                  } else {
                    if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.497866153717041238) ) ) {
                      result[0] += 0.007652824492584322;
                    } else {
                      result[0] += -0.06122247198290692;
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.53326439857482999) ) ) {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.012675821781158891) ) ) {
                  result[0] += 0.013146241635083884;
                } else {
                  result[0] += -0.011241670538328202;
                }
              } else {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += -0.029058148810236203;
                  } else {
                    result[0] += -0.000793136305702495;
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.040716171264650214) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                      result[0] += 0.027225695203055497;
                    } else {
                      result[0] += -0.0508416269901248;
                    }
                  } else {
                    result[0] += -0.056647686229255034;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.921100616455079013) ) ) {
              if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)10.8855171203613299) ) ) {
                if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)225.5000000000000284) ) ) {
                  if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                      result[0] += 0.03979109040524333;
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
                        result[0] += 0.03133392967205197;
                      } else {
                        result[0] += -0.05862984745457039;
                      }
                    }
                  } else {
                    result[0] += 0.013640725730170129;
                  }
                } else {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.019976781147650547;
                  } else {
                    result[0] += -0.00552139322142229;
                  }
                }
              } else {
                result[0] += -0.033773949005079684;
              }
            } else {
              if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)7.817222595214844638) ) ) {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.02731541889135898;
                } else {
                  result[0] += 0.025251582895611226;
                }
              } else {
                result[0] += -0.03689977547303049;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.591613531112671787) ) ) {
            result[0] += 0.00917723855701364;
          } else {
            result[0] += -0.009737789473377015;
          }
        } else {
          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += -0.05134697214426895;
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.921060562133789951) ) ) {
              result[0] += -0.006780859551011889;
            } else {
              result[0] += -0.03190895750420359;
            }
          }
        }
      }
    } else {
      result[0] += -0.0007278842938527156;
    }
  } else {
    if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.102609157562256748) ) ) {
      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.51693725585937678) ) ) {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.539549827575684482) ) ) {
          result[0] += -0.0005687023672097986;
        } else {
          result[0] += 0.006920124588543086;
        }
      } else {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += 0.013633539348640054;
        } else {
          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)197.5000000000000284) ) ) {
            if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
              result[0] += -0.015579402013641917;
            } else {
              if ( UNLIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += 0.029254901560317917;
              } else {
                if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.53326439857482999) ) ) {
                    result[0] += -0.047347804637371906;
                  } else {
                    result[0] += 0.013413388703010669;
                  }
                } else {
                  result[0] += -0.09347959825487934;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += -0.009847208001084538;
            } else {
              result[0] += -0.047487414435829885;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.553655147552491123) ) ) {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[60].missing != -1) || (data[60].fvalue <= (double)6.000000000000000888) ) ) {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.219419956207276279) ) ) {
              result[0] += -0.004692035914652012;
            } else {
              result[0] += -0.06727623088333483;
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
              result[0] += -0.02357035824724755;
            } else {
              result[0] += 0.02904895418170253;
            }
          }
        } else {
          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
            result[0] += -0.020772187099409728;
          } else {
            result[0] += -0.10869026701324165;
          }
        }
      } else {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.040618419647218573) ) ) {
            if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
              result[0] += -0.005803042070718488;
            } else {
              result[0] += -0.05731690537485398;
            }
          } else {
            result[0] += 0.014532106943747628;
          }
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.537586927413940874) ) ) {
            if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += 0.012432398448506188;
            } else {
              if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.002615194749053919;
                } else {
                  result[0] += -0.03262766303475272;
                }
              } else {
                result[0] += -0.07952653898639173;
              }
            }
          } else {
            result[0] += 0.021185182401368727;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
    if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)76.50000000000001421) ) ) {
      result[0] += 0.00010298596784174307;
    } else {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)5.500000000000000888) ) ) {
        if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)13.1100659370422381) ) ) {
          if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)136.5000000000000284) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.761470437049866167) ) ) {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.28299736976623624) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)14.20046806335449396) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.262283086776734287) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.01800251007080256) ) ) {
                        result[0] += -0.06063924551195458;
                      } else {
                        result[0] += -0.015000392903885254;
                      }
                    } else {
                      result[0] += 0.00028457091837999287;
                    }
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
                      result[0] += 0.10021656947875851;
                    } else {
                      result[0] += -0.01718253324822385;
                    }
                  }
                } else {
                  result[0] += -0.046682690337668614;
                }
              } else {
                if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)9.285166740417482245) ) ) {
                  result[0] += -0.0021533381627806274;
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += 0.08675832285810119;
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.131699204444885698) ) ) {
                      if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.15432071685791193) ) ) {
                          result[0] += -0.2175430091926591;
                        } else {
                          result[0] += 0.10229097727539907;
                        }
                      } else {
                        result[0] += -0.1273889468915886;
                      }
                    } else {
                      result[0] += -0.02570262161299664;
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.51517200469970881) ) ) {
                result[0] += -0.02108110824234135;
              } else {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.05881792284608263;
                } else {
                  if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)2.393745899200439897) ) ) {
                    result[0] += 0.05311965093779802;
                  } else {
                    result[0] += -0.1277459390370671;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
              if ( LIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += 0.0028841831794336565;
              } else {
                result[0] += -0.007707080182755959;
              }
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.397998809814454013) ) ) {
                    result[0] += 0.010195043449561235;
                  } else {
                    if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                      result[0] += -0.04000381594587349;
                    } else {
                      result[0] += 0.00022516117141828742;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
                      result[0] += -0.0037440259797303903;
                    } else {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.131699204444885698) ) ) {
                        result[0] += -0.00031625128244603187;
                      } else {
                        if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                          result[0] += -0.07014793290925787;
                        } else {
                          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
                              result[0] += -0.006546011895566144;
                            } else {
                              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                                result[0] += -0.06070894937774052;
                              } else {
                                result[0] += -0.018609004860259105;
                              }
                            }
                          } else {
                            result[0] += 0.004426124640653868;
                          }
                        }
                      }
                    }
                  } else {
                    result[0] += 0.0036858060092107127;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.512576580047609198) ) ) {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.174569487571716753) ) ) {
                    if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)219.5000000000000284) ) ) {
                      result[0] += 0.010035735566139889;
                    } else {
                      result[0] += -0.029420474507178596;
                    }
                  } else {
                    result[0] += 0.05602314942699264;
                  }
                } else {
                  result[0] += -0.04279831359023153;
                }
              }
            }
          }
        } else {
          result[0] += 0.02795627636004347;
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.58491539955139249) ) ) {
          result[0] += 0.009102828482920317;
        } else {
          result[0] += -0.05989356639676337;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)79.50000000000001421) ) ) {
      if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
        if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)24.00000000000000355) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.029068946838379794) ) ) {
            result[0] += -0.009853880487096567;
          } else {
            result[0] += -0.05291933935607101;
          }
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
            if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += 0.015557725566411565;
            } else {
              result[0] += -0.0034034484687976724;
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.241523027420044833) ) ) {
              result[0] += 0.013051694338520948;
            } else {
              result[0] += -0.008261205583673586;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)60.50000000000000711) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.0925779342651385) ) ) {
            result[0] += -0.003271048650171562;
          } else {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.017180835582844823;
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                result[0] += -0.029362782704606995;
              } else {
                if ( UNLIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += 0.039447136943899286;
                } else {
                  result[0] += -0.006816112362908727;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
              if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.666320323944092685) ) ) {
                result[0] += 0.00969272974107465;
              } else {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.693369150161744052) ) ) {
                      result[0] += 0.008706703963826416;
                    } else {
                      result[0] += -0.03913744378798782;
                    }
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.802696108818054643) ) ) {
                      result[0] += -0.010754344194481494;
                    } else {
                      result[0] += -0.07146912690474393;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.497866153717041238) ) ) {
                    result[0] += 0.01705321444810358;
                  } else {
                    if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += 0.019999926288148732;
                    } else {
                      result[0] += -0.05290621681579737;
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.20127439498901545) ) ) {
                result[0] += -0.02032896234909077;
              } else {
                result[0] += -0.07184830479162661;
              }
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
              result[0] += -0.04645270953353241;
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += -0.057730929430442036;
              } else {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
                  result[0] += 0.03259090059209784;
                } else {
                  result[0] += -0.01195326120038351;
                }
              }
            }
          }
        }
      }
    } else {
      result[0] += 0.0014298104647603102;
    }
  }
  if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
    if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
      if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)4.808102607727051669) ) ) {
        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)0.8958797454833985485) ) ) {
            result[0] += 0.07794107738775397;
          } else {
            result[0] += 0.013129199469975737;
          }
        } else {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
            result[0] += -0.0099626965617374;
          } else {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.384830474853516513) ) ) {
              result[0] += 0.01580425123613836;
            } else {
              result[0] += -0.03628840219859663;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.594915628433228427) ) ) {
            result[0] += -0.00691222397005764;
          } else {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.349750161170959917) ) ) {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                  result[0] += -0.06950989063230702;
                } else {
                  result[0] += -0.011051258104945729;
                }
              } else {
                if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += -0.07119610721461617;
                } else {
                  result[0] += -0.03417494618285015;
                }
              }
            } else {
              if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)10.50000000000000178) ) ) {
                  result[0] += -0.05211885450020111;
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.637949228286744052) ) ) {
                    result[0] += 0.013453058769639512;
                  } else {
                    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.03467573103867574;
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
                        result[0] += 0.2261806578819623;
                      } else {
                        result[0] += 0.07744526146782608;
                      }
                    }
                  }
                }
              } else {
                result[0] += -0.02608200626672464;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.262283086776734287) ) ) {
            result[0] += -0.03163583873945621;
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.775349855422974521) ) ) {
              result[0] += -0.01997530900768216;
            } else {
              if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.025842927369098673;
              } else {
                result[0] += -0.017411906239809663;
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)65.50000000000001421) ) ) {
        if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.553712725639343706) ) ) {
            result[0] += -0.0008497057345777883;
          } else {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                  if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)34.50000000000000711) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.142630577087403232) ) ) {
                      if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
                        result[0] += -0.09887113085591383;
                      } else {
                        result[0] += 0.0027613328102550593;
                      }
                    } else {
                      result[0] += 0.022449480132009106;
                    }
                  } else {
                    result[0] += 0.04060531448846108;
                  }
                } else {
                  if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.017421532079494133;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.587308406829834873) ) ) {
                      result[0] += -0.04085635904117768;
                    } else {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.94957673549652144) ) ) {
                        result[0] += -0.05118689635900989;
                      } else {
                        result[0] += 0.058559304761016784;
                      }
                    }
                  }
                }
              } else {
                result[0] += -0.03772838247758064;
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.241523027420044833) ) ) {
                result[0] += 0.018427454869750858;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.152389049530031073) ) ) {
                  result[0] += 0.0028921109774675546;
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
                    result[0] += -0.058669585974285836;
                  } else {
                    result[0] += -0.0035460010650100026;
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.835998296737671787) ) ) {
            result[0] += -0.017223210708692763;
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.64616632461548029) ) ) {
              result[0] += -0.0007957923194971764;
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.342454433441162998) ) ) {
                result[0] += -0.06982567537637151;
              } else {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                  result[0] += 0.07418628528219638;
                } else {
                  result[0] += -0.05054831204312099;
                }
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)2.764714598655701128) ) ) {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.700598716735840066) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.1746091842651385) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.154959201812744585) ) ) {
                result[0] += 0.015253068498719;
              } else {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)177.5000000000000284) ) ) {
                    result[0] += -0.03791947462565537;
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.071567356586456743) ) ) {
                      result[0] += 0.055284656967438776;
                    } else {
                      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
                        result[0] += -0.007326974390911381;
                      } else {
                        result[0] += -0.06313906131073742;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.094205617904663974) ) ) {
                    result[0] += -0.008503774262747971;
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                      if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.03420138359069913) ) ) {
                        result[0] += 0.008534321027607508;
                      } else {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.835998296737671787) ) ) {
                          result[0] += 0.0704002907718357;
                        } else {
                          result[0] += 0.009619311087407621;
                        }
                      }
                    } else {
                      result[0] += -0.007799164619629942;
                    }
                  }
                }
              }
            } else {
              result[0] += -0.024596960081793125;
            }
          } else {
            if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.700598716735840066) ) ) {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.497866153717041238) ) ) {
                  result[0] += -0.002005511390420723;
                } else {
                  if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                      result[0] += -0.056292536000927866;
                    } else {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.802696108818054643) ) ) {
                        result[0] += 0.029738507497033724;
                      } else {
                        result[0] += -0.04622602019917944;
                      }
                    }
                  } else {
                    result[0] += 0.0077732562309552646;
                  }
                }
              } else {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.802901029586792436) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.322819471359253818) ) ) {
                    result[0] += -0.03493599123485577;
                  } else {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)5.500000000000000888) ) ) {
                      result[0] += -0.006400389113943972;
                    } else {
                      result[0] += -0.0466798381198967;
                    }
                  }
                } else {
                  result[0] += 0.04929787428261486;
                }
              }
            } else {
              if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                result[0] += -0.06675322854072817;
              } else {
                result[0] += 0.07102872300922951;
              }
            }
          }
        } else {
          result[0] += 0.07063080558775424;
        }
      }
    }
  } else {
    result[0] += 0.0002291782908439593;
  }
  if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
    if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)76.50000000000001421) ) ) {
      if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
        result[0] += -0.0010986463319645544;
      } else {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
          if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.216319084167481357) ) ) {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.795426130294800249) ) ) {
                  if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)10.47656917572021662) ) ) {
                    if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)63.50000000000000711) ) ) {
                      result[0] += -0.06589681955760987;
                    } else {
                      result[0] += -0.016404888663910957;
                    }
                  } else {
                    result[0] += 0.0062909709710626096;
                  }
                } else {
                  result[0] += 0.013054792172865985;
                }
              } else {
                if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.987706661224366123) ) ) {
                  result[0] += -0.005317602924912203;
                } else {
                  result[0] += 0.028465364578111586;
                }
              }
            } else {
              if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.987706661224366123) ) ) {
                result[0] += -0.01723071161694523;
              } else {
                result[0] += -0.06417654491403066;
              }
            }
          } else {
            result[0] += 0.006890916255667329;
          }
        } else {
          if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
            result[0] += 0.019706758683620193;
          } else {
            if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += 0.029982793784953218;
            } else {
              result[0] += -0.010139216323674193;
            }
          }
        }
      }
    } else {
      result[0] += -0.001913226238013046;
    }
  } else {
    if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)79.50000000000001421) ) ) {
      if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
        if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)24.00000000000000355) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.029068946838379794) ) ) {
            result[0] += -0.0068509200725293574;
          } else {
            result[0] += -0.05017112721538534;
          }
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.993164777755738193) ) ) {
            result[0] += 4.093422000657613e-05;
          } else {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.53326439857482999) ) ) {
              if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += 0.00013403076478272274;
              } else {
                result[0] += 0.05787219723242827;
              }
            } else {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5.078289031982422763) ) ) {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.047021445105279114;
                  } else {
                    result[0] += -0.00727853778874264;
                  }
                } else {
                  result[0] += 0.04289529791759406;
                }
              } else {
                result[0] += 0.01858498489030446;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)60.50000000000000711) ) ) {
          result[0] += -0.0007911981114316092;
        } else {
          if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.666320323944092685) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.141444921493531162) ) ) {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.863673448562622958) ) ) {
                  result[0] += 0.012073682096500594;
                } else {
                  result[0] += -0.022900201297086534;
                }
              } else {
                result[0] += -0.01806907985563023;
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.029068946838379794) ) ) {
                result[0] += -0.0042177383867942125;
              } else {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.04378006395593648;
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.851041555404663974) ) ) {
                      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                        result[0] += 0.03076964289042526;
                      } else {
                        result[0] += -0.0804096437058008;
                      }
                    } else {
                      result[0] += -0.03916938512895238;
                    }
                  }
                } else {
                  result[0] += -0.05806961948261052;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
              result[0] += -0.04091888416515677;
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += -0.05245985449118119;
              } else {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
                  result[0] += 0.027652655272037488;
                } else {
                  result[0] += -0.01173298236611247;
                }
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)87.50000000000001421) ) ) {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.855006217956543857) ) ) {
          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.070700883865357333) ) ) {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += -0.0028538062430827826;
              } else {
                if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.736373662948609287) ) ) {
                  result[0] += 0.04507148951172745;
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.835998296737671787) ) ) {
                    result[0] += 0.030304528764986405;
                  } else {
                    result[0] += -0.05454199970559531;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.035777689352445685;
              } else {
                result[0] += -0.03326129748073616;
              }
            }
          } else {
            if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.497866153717041238) ) ) {
              result[0] += 0.04745999298443737;
            } else {
              result[0] += -0.004355313298660349;
            }
          }
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.843275547027588779) ) ) {
              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.802901029586792436) ) ) {
                result[0] += 0.06013005003771998;
              } else {
                result[0] += 0.0005570247072832701;
              }
            } else {
              result[0] += -0.020056108716424225;
            }
          } else {
            result[0] += -0.05223067014938193;
          }
        }
      } else {
        if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.012675821781158891) ) ) {
          if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += 0.019661418966130467;
              } else {
                result[0] += 0.08912550905541267;
              }
            } else {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
                result[0] += 0.00700709415027406;
              } else {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.198464870452881303) ) ) {
                  if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += -0.07799477538160393;
                  } else {
                    result[0] += -0.015025490839788902;
                  }
                } else {
                  if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                    if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += -0.009743689491659796;
                    } else {
                      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += 0.07876120106449591;
                      } else {
                        result[0] += 0.0031949898886388817;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                      result[0] += 0.0028179754764035736;
                    } else {
                      result[0] += -0.03501864401149266;
                    }
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += 0.014072558463884522;
            } else {
              result[0] += -0.04424997714171626;
            }
          }
        } else {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)5.500000000000000888) ) ) {
            result[0] += 0.0006204227430812996;
          } else {
            result[0] += -0.0672538698823259;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)6.500000000000000888) ) ) {
      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)8.285748958587648261) ) ) {
        result[0] += -0.0007718500629592508;
      } else {
        result[0] += -0.04090157854393241;
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.972535848617554599) ) ) {
        if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.53326439857482999) ) ) {
            result[0] += 0.0053762544842502315;
          } else {
            result[0] += -0.08179609733716438;
          }
        } else {
          result[0] += 0.021279400090181852;
        }
      } else {
        result[0] += -0.06878120352908344;
      }
    }
  } else {
    if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)79.50000000000001421) ) ) {
      if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
        if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)24.00000000000000355) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.029068946838379794) ) ) {
            result[0] += -0.006329372720778149;
          } else {
            result[0] += -0.04772423182940251;
          }
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
            result[0] += 0.00013271162178165075;
          } else {
            if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.318498134613038886) ) ) {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.674522399902344638) ) ) {
                  result[0] += -0.06688221885827061;
                } else {
                  result[0] += -0.00480313838288203;
                }
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.873467922210695136) ) ) {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += 0.03107533780260839;
                  } else {
                    result[0] += 0.002046807125291368;
                  }
                } else {
                  result[0] += -0.023343598581696483;
                }
              }
            } else {
              result[0] += 0.029914450212940175;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)60.50000000000000711) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.0925779342651385) ) ) {
            if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5.424940347671509677) ) ) {
              result[0] += -0.002996736464470749;
            } else {
              result[0] += -0.16672764711740407;
            }
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.802696108818054643) ) ) {
                if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += 0.032806946166347044;
                } else {
                  result[0] += -0.032175361647795594;
                }
              } else {
                result[0] += 0.016268053791961386;
              }
            } else {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.497866153717041238) ) ) {
                if ( UNLIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)24.50000000000000355) ) ) {
                    result[0] += 0.06731951099449444;
                  } else {
                    result[0] += -0.024661806831534558;
                  }
                } else {
                  result[0] += -0.05514981850775356;
                }
              } else {
                result[0] += 0.09453057695280356;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.363266706466675693) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.158952236175537998) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.055496215820313388) ) ) {
                  if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)71.50000000000001421) ) ) {
                    result[0] += -0.02962833675328684;
                  } else {
                    result[0] += 0.02795864466268734;
                  }
                } else {
                  result[0] += 0.1029777362472632;
                }
              } else {
                result[0] += 0.00020693653588722168;
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.53326439857482999) ) ) {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                  result[0] += 0.011044690574385051;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.040618419647218573) ) ) {
                    result[0] += 0.044241319292813;
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.802696108818054643) ) ) {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
                        result[0] += -0.03380622528686724;
                      } else {
                        result[0] += 0.00971343838646692;
                      }
                    } else {
                      result[0] += -0.04164053527383794;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.397998809814454013) ) ) {
                    result[0] += 0.018338730926261728;
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.88024568557739435) ) ) {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.944020271301270419) ) ) {
                        if ( UNLIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                          result[0] += -0.03197084475735259;
                        } else {
                          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.329314231872559482) ) ) {
                            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                              result[0] += 0.04272298754279777;
                            } else {
                              result[0] += -0.06523178233824965;
                            }
                          } else {
                            result[0] += -0.03791174259614428;
                          }
                        }
                      } else {
                        result[0] += -0.07016297455274514;
                      }
                    } else {
                      result[0] += -0.06846613154521902;
                    }
                  }
                } else {
                  result[0] += -0.0636337345923406;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
              result[0] += -0.03621968332084253;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.932935476303101474) ) ) {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                  result[0] += -0.04995688210815314;
                } else {
                  result[0] += 0.054006164648351375;
                }
              } else {
                result[0] += 0.00940835288837981;
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)87.50000000000001421) ) ) {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.855006217956543857) ) ) {
          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.070700883865357333) ) ) {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.556798219680787021) ) ) {
                  result[0] += -0.04426218338724705;
                } else {
                  result[0] += 0.0008786994516222743;
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.289602279663086826) ) ) {
                  result[0] += 0.05247907369945468;
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.835998296737671787) ) ) {
                    result[0] += 0.03254643202948659;
                  } else {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                      if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.068990230560303623) ) ) {
                        result[0] += 0.03014197021339557;
                      } else {
                        result[0] += -0.056057760606171195;
                      }
                    } else {
                      result[0] += -0.07813340974449859;
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.03548390605521874;
              } else {
                result[0] += -0.030918351215761933;
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.88024568557739435) ) ) {
              result[0] += 0.05996733422597805;
            } else {
              result[0] += 0.003260978090695039;
            }
          }
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.843275547027588779) ) ) {
              result[0] += 0.01759347331066442;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.36986422538757413) ) ) {
                result[0] += 0.03321326186895803;
              } else {
                if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.44140100479126021) ) ) {
                    result[0] += 0.007702665086130815;
                  } else {
                    result[0] += -0.06284928747521043;
                  }
                } else {
                  result[0] += -0.062479538286492164;
                }
              }
            }
          } else {
            result[0] += -0.04827026579425961;
          }
        }
      } else {
        result[0] += 0.0009500445916730745;
      }
    }
  }
  if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
    if ( UNLIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)1.00000001800250948e-35) ) ) {
      result[0] += 0.09081470090486221;
    } else {
      if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.285166740417482245) ) ) {
          result[0] += -0.08710229895778963;
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.51693725585937678) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.510617971420288974) ) ) {
              result[0] += -0.08618292871278305;
            } else {
              result[0] += -0.016296711022767748;
            }
          } else {
            result[0] += 0.009529257801904605;
          }
        }
      } else {
        if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)318.5000000000000568) ) ) {
          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)316.5000000000000568) ) ) {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)64.50000000000001421) ) ) {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)63.50000000000000711) ) ) {
                  result[0] += 0.0004995697991268242;
                } else {
                  if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.020127415657043901) ) ) {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.513699531555176669) ) ) {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.69067406654357999) ) ) {
                          result[0] += 0.04440785564610206;
                        } else {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.605120182037354404) ) ) {
                            result[0] += -0.12525980344341492;
                          } else {
                            result[0] += -0.03126432811276347;
                          }
                        }
                      } else {
                        result[0] += 0.017235110664941235;
                      }
                    } else {
                      result[0] += 0.03285024431152125;
                    }
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.531673669815064365) ) ) {
                      result[0] += -0.028216568691622826;
                    } else {
                      result[0] += 0.03668364020439986;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
                  if ( LIKELY( !(data[60].missing != -1) || (data[60].fvalue <= (double)6.000000000000000888) ) ) {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.617236852645874912) ) ) {
                      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.329314231872559482) ) ) {
                            result[0] += -0.023810432907084376;
                          } else {
                            result[0] += -0.0894847155517034;
                          }
                        } else {
                          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.617236852645874912) ) ) {
                            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.842459201812745917) ) ) {
                              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.497866153717041238) ) ) {
                                if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1.700598716735840066) ) ) {
                                  result[0] += 0.08369583410747344;
                                } else {
                                  result[0] += -0.023474434534167185;
                                }
                              } else {
                                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.216319084167481357) ) ) {
                                  result[0] += -0.06631808519765854;
                                } else {
                                  result[0] += -0.0072778639747347;
                                }
                              }
                            } else {
                              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.778982400894165927) ) ) {
                                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                                  result[0] += -0.00880628602410887;
                                } else {
                                  result[0] += -0.12487291150047508;
                                }
                              } else {
                                if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                                  result[0] += 0.004567765225116811;
                                } else {
                                  result[0] += 0.028087428470636312;
                                }
                              }
                            }
                          } else {
                            if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.067782521247864214) ) ) {
                              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.581200122833253729) ) ) {
                                result[0] += -0.03288320661192178;
                              } else {
                                result[0] += 0.07975065249606424;
                              }
                            } else {
                              result[0] += 0.071360082500455;
                            }
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.765202045440675604) ) ) {
                          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.924581527709961826) ) ) {
                            if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.676220536231995073) ) ) {
                              result[0] += -0.0019166992175228199;
                            } else {
                              result[0] += 0.03101123847808623;
                            }
                          } else {
                            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                              result[0] += -0.020945097676969122;
                            } else {
                              if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.020127415657043901) ) ) {
                                result[0] += 0.014242488468092233;
                              } else {
                                result[0] += -0.09590479924832804;
                              }
                            }
                          }
                        } else {
                          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.718933820724488193) ) ) {
                            if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.497866153717041238) ) ) {
                              result[0] += -0.07626390788197501;
                            } else {
                              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.531673669815064365) ) ) {
                                result[0] += -0.03405243336591853;
                              } else {
                                result[0] += 0.041735840995440594;
                              }
                            }
                          } else {
                            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                              result[0] += 0.003241057054310706;
                            } else {
                              result[0] += -0.0490541938814648;
                            }
                          }
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.632926940917970526) ) ) {
                          result[0] += -0.08242134350428203;
                        } else {
                          result[0] += 0.008062608219589743;
                        }
                      } else {
                        result[0] += 0.03589989849492331;
                      }
                    }
                  } else {
                    result[0] += 0.011601734846846056;
                  }
                } else {
                  result[0] += -0.05430243859542806;
                }
              }
            } else {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.539540290832521308) ) ) {
                  result[0] += -0.000641896717440558;
                } else {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                    if ( UNLIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += -0.01161929948707205;
                    } else {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.553712725639343706) ) ) {
                        result[0] += 0.009191570462652484;
                      } else {
                        result[0] += -0.005340574595374335;
                      }
                    }
                  } else {
                    result[0] += -0.040323156951975886;
                  }
                }
              } else {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
                  result[0] += 0.0040436695667166005;
                } else {
                  result[0] += -0.0322342478965981;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.497866153717041238) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.126503944396974433) ) ) {
                if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                  result[0] += 0.05551606038341497;
                } else {
                  if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)3.134548187255859819) ) ) {
                    result[0] += 0.0429311666572112;
                  } else {
                    result[0] += -0.01040803682563004;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.142630577087403232) ) ) {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.740319490432739702) ) ) {
                    result[0] += 0.1164912331841675;
                  } else {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.067782521247864214) ) ) {
                      result[0] += -0.0394780450220606;
                    } else {
                      result[0] += 0.07250606605947195;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.662244915962219682) ) ) {
                    if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.014831542968751776) ) ) {
                      result[0] += -0.13800023085755597;
                    } else {
                      result[0] += 0.005440355031140435;
                    }
                  } else {
                    result[0] += 0.023206364495984805;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.764287948608400214) ) ) {
                result[0] += -0.12439361605817856;
              } else {
                result[0] += 0.006259019785380753;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.08150768280029475) ) ) {
            result[0] += -0.005464412404948795;
          } else {
            if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += -0.05399188721152392;
            } else {
              result[0] += 0.09047793947720012;
            }
          }
        }
      }
    }
  } else {
    result[0] += 0.0007386276224264766;
  }
  if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
    if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)64.50000000000001421) ) ) {
      if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.142630577087403232) ) ) {
          result[0] += -0.029387128775880594;
        } else {
          if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.085941076278687412) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.467917680740357333) ) ) {
              if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                result[0] += 0.01820695818883209;
              } else {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.10550346695531916;
                } else {
                  if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.44140100479126021) ) ) {
                    result[0] += -0.06500403050337435;
                  } else {
                    result[0] += -0.012449483599673707;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                  if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.497866153717041238) ) ) {
                    result[0] += -0.19220536346996628;
                  } else {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.249904870986938921) ) ) {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.851041555404663974) ) ) {
                        if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.603265762329102451) ) ) {
                          result[0] += -0.06464444594761627;
                        } else {
                          result[0] += -0.00015599803294034996;
                        }
                      } else {
                        if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.020127415657043901) ) ) {
                          result[0] += -0.005526685289477704;
                        } else {
                          result[0] += 0.04740720000087709;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                        result[0] += 0.05984989920269105;
                      } else {
                        result[0] += 0.013402075515888812;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.802901029586792436) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.921060562133789951) ) ) {
                      result[0] += -0.021738350638386198;
                    } else {
                      result[0] += 0.037377263784533404;
                    }
                  } else {
                    result[0] += 0.06209408926375104;
                  }
                }
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.617236852645874912) ) ) {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.011018627151133872;
                    } else {
                      result[0] += 0.04092274742385618;
                    }
                  } else {
                    if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.078787803649903232) ) ) {
                      result[0] += -0.0013150038122215924;
                    } else {
                      result[0] += -0.05976508451593511;
                    }
                  }
                } else {
                  result[0] += 0.029088448017922975;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
              if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.835998296737671787) ) ) {
                  if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)10.47656917572021662) ) ) {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.21812704821457152;
                    } else {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
                        result[0] += -0.0534527243540506;
                      } else {
                        result[0] += 0.06391570068417451;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.05847201962946557;
                    } else {
                      result[0] += -0.09012396080644539;
                    }
                  }
                } else {
                  result[0] += 0.05294481315983162;
                }
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
                  if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.650573849678039995) ) ) {
                    result[0] += -0.19516939941001635;
                  } else {
                    if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.497866153717041238) ) ) {
                      result[0] += -0.024904112105182237;
                    } else {
                      result[0] += 0.05717499283156061;
                    }
                  }
                } else {
                  result[0] += 0.039757349488294146;
                }
              }
            } else {
              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.700598716735840066) ) ) {
                if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)9.285166740417482245) ) ) {
                  result[0] += -0.005590350601920846;
                } else {
                  result[0] += -0.07426173387185184;
                }
              } else {
                result[0] += 0.08699096931216438;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.921100616455079013) ) ) {
            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
                if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.433743238449097568) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.450390577316285068) ) ) {
                    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.019594546139193838;
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.322819471359253818) ) ) {
                        result[0] += 0.015466233558076312;
                      } else {
                        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.745876312255860263) ) ) {
                            result[0] += -0.04511507535927103;
                          } else {
                            result[0] += -0.006059807312956168;
                          }
                        } else {
                          result[0] += 0.008276106595967345;
                        }
                      }
                    }
                  } else {
                    result[0] += -0.05110635298090682;
                  }
                } else {
                  result[0] += 0.0013724498768715573;
                }
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.795762062072754794) ) ) {
                  if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)7.690742254257203037) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.637949228286744052) ) ) {
                      if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)4.749261140823365146) ) ) {
                        result[0] += 0.06391580213218877;
                      } else {
                        result[0] += -0.033089824842019964;
                      }
                    } else {
                      result[0] += 0.03370103627497719;
                    }
                  } else {
                    result[0] += -0.021888046143762954;
                  }
                } else {
                  result[0] += -0.03356493202903615;
                }
              }
            } else {
              if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += 0.010457160081767308;
              } else {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)20.50000000000000355) ) ) {
                  result[0] += -0.040653714549391255;
                } else {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.53326439857482999) ) ) {
                      result[0] += -0.034836738969241915;
                    } else {
                      if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.399426221847535068) ) ) {
                        result[0] += -0.06583683955613116;
                      } else {
                        result[0] += 0.015889270262757438;
                      }
                    }
                  } else {
                    result[0] += -0.025308301542848518;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.07025824522612231;
            } else {
              result[0] += -0.004721404524401648;
            }
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.397998809814454013) ) ) {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
              if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.005538647030301724;
              } else {
                result[0] += -0.009308723758787546;
              }
            } else {
              if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.269673109054566318) ) ) {
                result[0] += -0.038308085512267526;
              } else {
                result[0] += -0.11466918785309826;
              }
            }
          } else {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += 0.0019483739994065686;
            } else {
              if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.424828529357911044) ) ) {
                result[0] += 0.008919305919238915;
              } else {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.10390568802954857;
                } else {
                  result[0] += 0.03277257693920126;
                }
              }
            }
          }
        }
      }
    } else {
      result[0] += -0.0013733444136065037;
    }
  } else {
    result[0] += 0.0007485526751303047;
  }
  if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
    if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.182021141052246982) ) ) {
      if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)6144.000000000000909) ) ) {
        if ( UNLIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
          if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.472188472747804511) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
              if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.605120182037354404) ) ) {
                  result[0] += 0.05726981251911472;
                } else {
                  result[0] += -0.007536646674357581;
                }
              } else {
                result[0] += 0.013956069937255412;
              }
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.790116786956788886) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.617236852645874912) ) ) {
                  result[0] += 0.0027369371563173853;
                } else {
                  result[0] += -0.029796102840868623;
                }
              } else {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)45.50000000000000711) ) ) {
                  result[0] += 0.005879614243409282;
                } else {
                  if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.017344867944262956;
                  } else {
                    result[0] += -0.07080771450256262;
                  }
                }
              }
            }
          } else {
            result[0] += -0.07303410378301642;
          }
        } else {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.216319084167481357) ) ) {
            if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)2.071567356586456743) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.04641723632812678) ) ) {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
                    if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
                      if ( UNLIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += -0.0031388130630994014;
                      } else {
                        if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)7.690742254257203037) ) ) {
                          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)1.700598716735840066) ) ) {
                            result[0] += 0.0009494819563460413;
                          } else {
                            result[0] += -0.04123274352681968;
                          }
                        } else {
                          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.737386107444763628) ) ) {
                            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                              result[0] += -0.026719894661568162;
                            } else {
                              result[0] += 0.0601883368297245;
                            }
                          } else {
                            result[0] += -0.04382536727593027;
                          }
                        }
                      }
                    } else {
                      result[0] += -0.03385059577689525;
                    }
                  } else {
                    result[0] += -0.05520009094427031;
                  }
                } else {
                  if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
                    result[0] += -0.008796307178753649;
                  } else {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.932935476303101474) ) ) {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.837713479995728427) ) ) {
                          result[0] += -0.024047328040143572;
                        } else {
                          result[0] += -0.08011774896861462;
                        }
                      } else {
                        result[0] += 0.00636812882252236;
                      }
                    } else {
                      if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.918272972106934482) ) ) {
                          result[0] += 0.03778946603044794;
                        } else {
                          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.837713479995728427) ) ) {
                            if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.951942920684815341) ) ) {
                              result[0] += 0.00233966344502053;
                            } else {
                              result[0] += -0.044958740820115424;
                            }
                          } else {
                            result[0] += 0.021169716160339115;
                          }
                        }
                      } else {
                        result[0] += -0.023050696835991397;
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += 0.001575824380256911;
                  } else {
                    result[0] += 0.027363419090029456;
                  }
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.777633190155030185) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.14301252365112482) ) ) {
                      result[0] += 0.0169869172916645;
                    } else {
                      result[0] += -0.03809610317503728;
                    }
                  } else {
                    result[0] += 0.02139382962229772;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.057859051020545706;
              } else {
                result[0] += 0.07078121465041441;
              }
            }
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)5.500000000000000888) ) ) {
              result[0] += 0.006125585145480498;
            } else {
              result[0] += -0.0578146680810462;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.802696108818054643) ) ) {
          result[0] += -0.0015612579797178337;
        } else {
          result[0] += -0.0354839354059024;
        }
      }
    } else {
      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
        if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.637949228286744052) ) ) {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.76779222488403498) ) ) {
                result[0] += 0.0035210350786862078;
              } else {
                if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                  result[0] += 0.008122064505643772;
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
                    result[0] += 0.0035568824246213445;
                  } else {
                    if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.12291239507095056;
                    } else {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.673553824424744096) ) ) {
                        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
                          result[0] += 0.06963259436957246;
                        } else {
                          result[0] += -0.0864876345433974;
                        }
                      } else {
                        result[0] += -0.07295975612498205;
                      }
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)6.000000000000000888) ) ) {
                if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += 0.012421289762374294;
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
                    result[0] += 0.01693580287366648;
                  } else {
                    result[0] += -0.05957593886712567;
                  }
                }
              } else {
                result[0] += 0.022244680782073287;
              }
            }
          } else {
            if ( UNLIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.0820959554902859;
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.8387184143066424) ) ) {
                if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.03728105635374185;
                } else {
                  result[0] += 0.014126286534920941;
                }
              } else {
                result[0] += 0.06729962022725108;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += -0.07831579212238939;
          } else {
            if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)2.764714598655701128) ) ) {
              result[0] += -0.024668381565247036;
            } else {
              result[0] += 0.07580453153934123;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.510617971420288974) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.1746091842651385) ) ) {
            result[0] += 0.04690543626200521;
          } else {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.342454433441162998) ) ) {
              result[0] += -0.04203855511349478;
            } else {
              result[0] += 0.030035693511031882;
            }
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.56941866874694913) ) ) {
            result[0] += -0.04787991819882209;
          } else {
            if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += -0.08908229695854661;
            } else {
              if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)20.50000000000000355) ) ) {
                result[0] += 0.025991445858732778;
              } else {
                result[0] += -0.023694054535100064;
              }
            }
          }
        }
      }
    }
  } else {
    result[0] += 0.0003068839181653667;
  }
  if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)6.000000000000000888) ) ) {
    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.72660112380981623) ) ) {
      if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.124530076980591708) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.967588424682618964) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.605120182037354404) ) ) {
              if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)5.500000000000000888) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.837713479995728427) ) ) {
                  if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)36.50000000000000711) ) ) {
                    result[0] += -0.039357832875444726;
                  } else {
                    result[0] += 0.04223102091149682;
                  }
                } else {
                  result[0] += -0.05961457011604737;
                }
              } else {
                result[0] += 0.07055220901020165;
              }
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.921100616455079013) ) ) {
                result[0] += -0.003616121777972095;
              } else {
                result[0] += -0.07450800512786923;
              }
            }
          } else {
            if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.863673448562622958) ) ) {
              result[0] += -0.03773871686093908;
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.241523027420044833) ) ) {
                result[0] += 0.011740514656048787;
              } else {
                result[0] += -0.06315967610507976;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.245136737823487216) ) ) {
              result[0] += -0.022826951145242923;
            } else {
              result[0] += -0.07103794952491094;
            }
          } else {
            if ( UNLIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.05828399867040157;
            } else {
              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.249904870986938921) ) ) {
                result[0] += 0.3261331088852528;
              } else {
                result[0] += 0.03705512864468998;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.10848760604858576) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
              if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)4.500000000000000888) ) ) {
                if ( UNLIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.04150080683481598;
                } else {
                  result[0] += 0.01438445749542055;
                }
              } else {
                result[0] += 0.07407101110046042;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.932935476303101474) ) ) {
                if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)36.50000000000000711) ) ) {
                  result[0] += -0.012187665216229562;
                } else {
                  result[0] += 0.004732799136262568;
                }
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.921100616455079013) ) ) {
                  if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                    if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += 0.06648099851582491;
                      } else {
                        result[0] += -0.023146732946674912;
                      }
                    } else {
                      result[0] += -0.05658538071711958;
                    }
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.329314231872559482) ) ) {
                      if ( UNLIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)2.500000000000000444) ) ) {
                        if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)4.500000000000000888) ) ) {
                          result[0] += -0.044775962190418295;
                        } else {
                          if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.648983716964722568) ) ) {
                            if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.238486170768738237) ) ) {
                              result[0] += 0.003945336444587494;
                            } else {
                              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.381086945533752885) ) ) {
                                result[0] += -0.10488161004884147;
                              } else {
                                result[0] += -0.02298429129635779;
                              }
                            }
                          } else {
                            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.156774044036865678) ) ) {
                              result[0] += 0.05840834783858249;
                            } else {
                              result[0] += -2.220640889515934e-05;
                            }
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.377930641174318183) ) ) {
                          result[0] += -0.038559028167424486;
                        } else {
                          result[0] += -0.10492739383288802;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)2.500000000000000444) ) ) {
                        result[0] += -0.053013618347367286;
                      } else {
                        if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)6.500000000000000888) ) ) {
                          if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                            result[0] += -0.013469984737903918;
                          } else {
                            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                              result[0] += 0.07361680758026892;
                            } else {
                              result[0] += 0.015865495690354395;
                            }
                          }
                        } else {
                          result[0] += -0.08791141313384727;
                        }
                      }
                    }
                  }
                } else {
                  result[0] += -0.07387485518771443;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.094205617904663974) ) ) {
                if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.04412649975834724;
                  } else {
                    result[0] += -0.07950501861232818;
                  }
                } else {
                  if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.855921268463135654) ) ) {
                    result[0] += 0.025333667662383392;
                  } else {
                    result[0] += 0.0919580337927659;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.03564921068647875;
                } else {
                  result[0] += -0.013139054347701951;
                }
              }
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.404924631118775302) ) ) {
                  if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.300811052322388583) ) ) {
                    result[0] += 0.1341975989773744;
                  } else {
                    result[0] += 0.6273087841081917;
                  }
                } else {
                  result[0] += 0.06239992400277066;
                }
              } else {
                result[0] += -0.03627535442738868;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += 0.014946429894468958;
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += -0.07173793906498498;
              } else {
                result[0] += 0.023701286473387453;
              }
            } else {
              result[0] += 0.005306478417236728;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.662244915962219682) ) ) {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
          if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
            result[0] += -0.05702747475496506;
          } else {
            if ( UNLIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.10508068517717604;
            } else {
              result[0] += -0.0062010661313528035;
            }
          }
        } else {
          if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)2.500000000000000444) ) ) {
            result[0] += -0.07720704703288989;
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.553655147552491123) ) ) {
              result[0] += -0.05755267061342291;
            } else {
              result[0] += 0.08874968409830139;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.94957673549652144) ) ) {
            result[0] += 0.025235418041229835;
          } else {
            result[0] += -0.08018821387961433;
          }
        } else {
          if ( UNLIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += -0.0006458406542956389;
            } else {
              result[0] += 0.05265751328807125;
            }
          } else {
            result[0] += -0.05129953703002668;
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)1.00000001800250948e-35) ) ) {
      result[0] += -0.014537217959753707;
    } else {
      result[0] += 0.00016154075502485346;
    }
  }
  if ( UNLIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.00000001800250948e-35) ) ) {
    result[0] += 0.08358176191726718;
  } else {
    if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.26837396621704279) ) ) {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.040618419647218573) ) ) {
          result[0] += -0.08483728225964296;
        } else {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.534971714019776279) ) ) {
            result[0] += -0.04149032887937073;
          } else {
            result[0] += 0.11310622807905531;
          }
        }
      } else {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.761470437049866167) ) ) {
            result[0] += 0.018013762638380778;
          } else {
            result[0] += -0.035807301625459094;
          }
        } else {
          result[0] += -0.03856781199342157;
        }
      }
    } else {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)6.500000000000000888) ) ) {
        if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)8.285748958587648261) ) ) {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)76.50000000000001421) ) ) {
              result[0] += 0.00035956717495959616;
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.868834793567657693) ) ) {
                if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.47529649734497248) ) ) {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.010734449590363902;
                    } else {
                      result[0] += -0.05656088303682519;
                    }
                  } else {
                    if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)83.50000000000001421) ) ) {
                      result[0] += 0.062247263048474846;
                    } else {
                      result[0] += 0.02627338335085699;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.569433569908142534) ) ) {
                    result[0] += -0.01582493907133798;
                  } else {
                    if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += 0.014896013916703699;
                    } else {
                      result[0] += -0.012063691400797182;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.363078355789185458) ) ) {
                  if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)136.5000000000000284) ) ) {
                    if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.28299736976623624) ) ) {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.262283086776734287) ) ) {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.14301252365112482) ) ) {
                            result[0] += -0.07342383611948598;
                          } else {
                            result[0] += -0.020529934643947687;
                          }
                        } else {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.152389049530031073) ) ) {
                            result[0] += -0.01377313305773233;
                          } else {
                            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.737386107444763628) ) ) {
                              result[0] += 0.06608653381909083;
                            } else {
                              result[0] += 0.006658186203963807;
                            }
                          }
                        }
                      } else {
                        result[0] += -0.042178123199912605;
                      }
                    } else {
                      result[0] += -0.0012891169421115284;
                    }
                  } else {
                    if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)137.5000000000000284) ) ) {
                      result[0] += 0.022194514155590088;
                    } else {
                      result[0] += -0.0003808769350538867;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[58].missing != -1) || (data[58].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)218.5000000000000284) ) ) {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                          if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                            result[0] += 0.003040215793956514;
                          } else {
                            if ( LIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)48.00000000000000711) ) ) {
                              result[0] += -0.04675515421902044;
                            } else {
                              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.04690074920654475) ) ) {
                                result[0] += -0.039606950293753496;
                              } else {
                                result[0] += 0.03975226227569145;
                              }
                            }
                          }
                        } else {
                          if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            result[0] += -0.07418461765064908;
                          } else {
                            result[0] += -0.022227618715367093;
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.497866153717041238) ) ) {
                            if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)308.5000000000000568) ) ) {
                              result[0] += 0.026549916606565683;
                            } else {
                              result[0] += -0.013088640643037412;
                            }
                          } else {
                            result[0] += -0.04419529569692636;
                          }
                        } else {
                          result[0] += -0.045881941526630984;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.861792564392090288) ) ) {
                        result[0] += -0.026465252935571892;
                      } else {
                        result[0] += -0.08139989248066802;
                      }
                    }
                  } else {
                    result[0] += 0.017713703524014298;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
              result[0] += -0.09876277608517499;
            } else {
              result[0] += -0.012529117370952414;
            }
          }
        } else {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)5.500000000000000888) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.761470437049866167) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.381086945533752885) ) ) {
                result[0] += 0.010297576566423959;
              } else {
                result[0] += 0.0006497529060502803;
              }
            } else {
              if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.623839378356934482) ) ) {
                    if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                      if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5.059420347213746005) ) ) {
                        result[0] += -0.12515839555589744;
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.89450073242187678) ) ) {
                          result[0] += -0.023470209154845403;
                        } else {
                          result[0] += 0.03706220564097032;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.722943305969239169) ) ) {
                        result[0] += 0.0005463276240594121;
                      } else {
                        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += -0.031781800195540236;
                        } else {
                          result[0] += 0.0333474270870994;
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.09734183561869998;
                    } else {
                      result[0] += 0.030181127610527228;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += -0.0018267099885105533;
                  } else {
                    result[0] += 0.1015337888020951;
                  }
                }
              } else {
                result[0] += 0.07120397388416193;
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.605120182037354404) ) ) {
              result[0] += 0.029964647628988823;
            } else {
              result[0] += -0.08221684599635547;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.58491539955139249) ) ) {
          if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)34.50000000000000711) ) ) {
              result[0] += -0.06649548155025765;
            } else {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += -0.016784322372791316;
              } else {
                result[0] += 0.06523881653366465;
              }
            }
          } else {
            result[0] += 0.09738221812372956;
          }
        } else {
          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.591613531112671787) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.213027238845826083) ) ) {
                result[0] += -0.02239429696112311;
              } else {
                result[0] += -0.08009357951719305;
              }
            } else {
              result[0] += -0.0912365477454326;
            }
          } else {
            if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)71.50000000000001421) ) ) {
              result[0] += 0.07127958669819437;
            } else {
              result[0] += -0.08938035788815459;
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
    result[0] += 0.07965829764246368;
  } else {
    if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)7.500000000000000888) ) ) {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.213027238845826083) ) ) {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
          if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)2.500000000000000444) ) ) {
            result[0] += -0.0654332820978884;
          } else {
            if ( LIKELY( !(data[56].missing != -1) || (data[56].fvalue <= (double)3.000000000000000444) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.071567356586456743) ) ) {
                result[0] += 0.008793773223339865;
              } else {
                result[0] += -0.03947697714088934;
              }
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.623839378356934482) ) ) {
                result[0] += 0.019517339089464913;
              } else {
                result[0] += -0.06959045307720027;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.272946834564209873) ) ) {
            result[0] += -0.028818578755807362;
          } else {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.552201986312867099) ) ) {
              if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.026451856093585586;
              } else {
                result[0] += -0.004099868061951478;
              }
            } else {
              result[0] += -0.06717330037339718;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
          if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
              if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.48298668861389249) ) ) {
                result[0] += 0.0011871159273527448;
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.53326439857482999) ) ) {
                  result[0] += 0.012060004055853474;
                } else {
                  result[0] += 0.11432681495396771;
                }
              }
            } else {
              if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.450390577316285068) ) ) {
                  result[0] += -0.08768047825335497;
                } else {
                  result[0] += -0.007376908982282474;
                }
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.947818994522095615) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.190353393554689276) ) ) {
                    result[0] += -0.08505521410492475;
                  } else {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.637949228286744052) ) ) {
                      result[0] += -0.0292580033008362;
                    } else {
                      if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += -0.06100605898098296;
                      } else {
                        result[0] += 0.010678348588451252;
                      }
                    }
                  }
                } else {
                  result[0] += 0.025554021048093862;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += -0.00270683807061029;
              } else {
                result[0] += -0.07916714925766763;
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.342454433441162998) ) ) {
                result[0] += -0.006098167011803062;
              } else {
                result[0] += 0.040184761996867935;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.868834793567657693) ) ) {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.693369150161744052) ) ) {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                  if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)7.933616161346436435) ) ) {
                    result[0] += -0.046512942223973085;
                  } else {
                    result[0] += 0.0009237261436773939;
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.737386107444763628) ) ) {
                    result[0] += -0.04433251288122025;
                  } else {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                      result[0] += -0.02187057532041511;
                    } else {
                      result[0] += 0.05805374908021382;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.04815528036305093;
                } else {
                  result[0] += -0.02354929366634874;
                }
              }
            } else {
              if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)4.500000000000000888) ) ) {
                result[0] += -0.029075279848466723;
              } else {
                if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.497866153717041238) ) ) {
                  if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)9.437634944915773261) ) ) {
                    result[0] += -0.014822133684447045;
                  } else {
                    result[0] += 0.05032593282892688;
                  }
                } else {
                  result[0] += -0.04704664148805101;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.637949228286744052) ) ) {
              if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.539299011230469638) ) ) {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.993822574615479404) ) ) {
                    result[0] += -0.06785658761579977;
                  } else {
                    result[0] += -0.008510278530219526;
                  }
                } else {
                  result[0] += 0.027312121358177784;
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.071567356586456743) ) ) {
                  result[0] += 0.08930955725409652;
                } else {
                  result[0] += 0.007766312913957978;
                }
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.636499762535095659) ) ) {
                result[0] += 0.11883144723145203;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.126503944396974433) ) ) {
                  result[0] += 0.003582168718391874;
                } else {
                  result[0] += 0.027573648695036923;
                }
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)8.500000000000001776) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.802696108818054643) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.967588424682618964) ) ) {
            result[0] += -0.012188558120701883;
          } else {
            if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += 0.10605036488560218;
            } else {
              if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
                result[0] += -0.043329466631898426;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.58713245391845881) ) ) {
                  result[0] += -0.02317040457519448;
                } else {
                  result[0] += 0.06526579730774971;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.67574596405029475) ) ) {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.53326439857482999) ) ) {
                  result[0] += -0.00047897286860282295;
                } else {
                  result[0] += -0.06978764763997572;
                }
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.617236852645874912) ) ) {
                  result[0] += 0.018765085426678336;
                } else {
                  result[0] += -0.07689416698842912;
                }
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.158952236175537998) ) ) {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.649621725082398349) ) ) {
                    result[0] += -0.04425543917387077;
                  } else {
                    result[0] += 0.032566656361994455;
                  }
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.815814018249513495) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.171656608581543857) ) ) {
                      result[0] += 0.06344577909728433;
                    } else {
                      result[0] += 0.12066704992891186;
                    }
                  } else {
                    result[0] += -0.017268641190267497;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += 0.033329469835794295;
                  } else {
                    result[0] += -0.019558795520871627;
                  }
                } else {
                  result[0] += -0.04656817942386149;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
              result[0] += -0.016423807901043443;
            } else {
              result[0] += -0.07560155602871076;
            }
          }
        }
      } else {
        result[0] += -5.68121923128792e-06;
      }
    }
  }
  if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)48.00000000000000711) ) ) {
    result[0] += 0.0003232773528780336;
  } else {
    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.300458192825318271) ) ) {
      if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)218.5000000000000284) ) ) {
        if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.579273939132691318) ) ) {
            if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.547126770019532138) ) ) {
              if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)6.000000000000000888) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.674522399902344638) ) ) {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.085941076278687412) ) ) {
                    result[0] += -0.016424766730508494;
                  } else {
                    result[0] += -0.08373757119037956;
                  }
                } else {
                  if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.067782521247864214) ) ) {
                      if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)0.8958797454833985485) ) ) {
                          result[0] += -0.16086028509853378;
                        } else {
                          result[0] += -0.014141811956931212;
                        }
                      } else {
                        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)1.868834793567657693) ) ) {
                          result[0] += 0.07103502827687228;
                        } else {
                          result[0] += 0.010467227233693388;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.03420138359069913) ) ) {
                        result[0] += -0.01899184095101622;
                      } else {
                        result[0] += 0.005676520420273589;
                      }
                    }
                  } else {
                    result[0] += -0.019658724336385346;
                  }
                }
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.06299877166748225) ) ) {
                  result[0] += -0.03091419650281825;
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)17.0776128768920934) ) ) {
                    result[0] += 0.00873835302191781;
                  } else {
                    result[0] += -0.1295894676259703;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.431901693344116655) ) ) {
                  if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)135.5000000000000284) ) ) {
                    result[0] += 0.01483567338123043;
                  } else {
                    result[0] += 0.0576990224480251;
                  }
                } else {
                  if ( LIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)12.00000000000000178) ) ) {
                    if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += 0.023644792085129665;
                      } else {
                        result[0] += -0.033631079056326424;
                      }
                    } else {
                      result[0] += 0.031950399989169695;
                    }
                  } else {
                    result[0] += -0.08211463118283077;
                  }
                }
              } else {
                result[0] += -0.036053796946051674;
              }
            }
          } else {
            if ( LIKELY( !(data[58].missing != -1) || (data[58].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.06682129054618693;
            } else {
              result[0] += -0.030120969135912903;
            }
          }
        } else {
          if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)6.000000000000000888) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.164715528488160068) ) ) {
                  result[0] += 0.007444604746459928;
                } else {
                  result[0] += -0.008614782545208092;
                }
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.94957673549652144) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.264029741287232333) ) ) {
                    result[0] += -0.03179518668308739;
                  } else {
                    result[0] += 0.013316249563166885;
                  }
                } else {
                  if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                    if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.349750161170959917) ) ) {
                      result[0] += 0.011898288633615835;
                    } else {
                      result[0] += 0.03971357947968563;
                    }
                  } else {
                    result[0] += 0.046225241896646375;
                  }
                }
              }
            } else {
              result[0] += -0.0016980814284451697;
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.94957673549652144) ) ) {
              result[0] += 0.01611727415828085;
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.13022470474243342) ) ) {
                result[0] += -0.011723612976998202;
              } else {
                result[0] += -0.04564198620341344;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.497866153717041238) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.60200452804565607) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.088880300521851474) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.901921629905701128) ) ) {
                  result[0] += 0.04076864226688797;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.439304351806642401) ) ) {
                    if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)1.242453336715698464) ) ) {
                        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
                          result[0] += -0.08843387309483086;
                        } else {
                          result[0] += 0.009801621716715843;
                        }
                      } else {
                        result[0] += -0.009545258276130324;
                      }
                    } else {
                      result[0] += 0.0024045842526948403;
                    }
                  } else {
                    result[0] += 0.01122171703698329;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.972535848617554599) ) ) {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.23832273483276456) ) ) {
                    result[0] += 0.00540724605066124;
                  } else {
                    result[0] += -0.04209053994158319;
                  }
                } else {
                  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
                    result[0] += 0.010831705776048726;
                  } else {
                    result[0] += 0.028850834345212496;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += 0.038060637378196094;
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.342454433441162998) ) ) {
                  result[0] += 0.03881049968492817;
                } else {
                  result[0] += -0.017728359243175937;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.51011085510254084) ) ) {
              result[0] += -0.051523936567692934;
            } else {
              result[0] += -0.004429877148718737;
            }
          }
        } else {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.13022470474243342) ) ) {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)221.5000000000000284) ) ) {
              result[0] += 0.01779038434556154;
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.623839378356934482) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.342454433441162998) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.662244915962219682) ) ) {
                    result[0] += 0.007994243241649835;
                  } else {
                    result[0] += -0.02887149979224948;
                  }
                } else {
                  if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += -0.013207261805652496;
                  } else {
                    result[0] += 0.005287008147179709;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.513969182968140537) ) ) {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                    result[0] += 0.0015197638689886475;
                  } else {
                    result[0] += -0.03109780625046479;
                  }
                } else {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.01968802025489998;
                  } else {
                    result[0] += -0.05457208718847907;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.662244915962219682) ) ) {
              if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.044808515701734414;
              } else {
                result[0] += -0.04348969578985485;
              }
            } else {
              if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.744894266128540927) ) ) {
                result[0] += -0.03840185631197527;
              } else {
                result[0] += -0.07971738058761653;
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
        result[0] += -0.03663205229312951;
      } else {
        result[0] += -0.0030225328616024557;
      }
    }
  }
  if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
    result[0] += -0.00025801541325031445;
  } else {
    if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.449861526489258257) ) ) {
      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.673553824424744096) ) ) {
        if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)156.5000000000000284) ) ) {
          if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.012675821781158891) ) ) {
            result[0] += -0.04949439099437354;
          } else {
            result[0] += 0.030442817492028408;
          }
        } else {
          result[0] += -0.004271995245314015;
        }
      } else {
        result[0] += -0.000615873558875787;
      }
    } else {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.53326439857482999) ) ) {
        if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)102.5000000000000142) ) ) {
          result[0] += -0.01932985886606281;
        } else {
          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
            result[0] += 0.004062137082193633;
          } else {
            if ( UNLIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.89450073242187678) ) ) {
                result[0] += -0.0047103419123562485;
              } else {
                result[0] += 0.021887818803300388;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.54081821441650568) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.510617971420288974) ) ) {
                  result[0] += 0.024315518835205926;
                } else {
                  result[0] += -0.03037919418239507;
                }
              } else {
                result[0] += -0.04026571640284204;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)2.500000000000000444) ) ) {
          if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
            result[0] += 0.0028923024810038026;
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.321723937988282138) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.909102678298951083) ) ) {
                result[0] += -0.051214363862325564;
              } else {
                if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                  if ( UNLIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
                        result[0] += 0.0020057101981523474;
                      } else {
                        result[0] += 0.03854265037277335;
                      }
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.54081821441650568) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.711936950683595526) ) ) {
                          result[0] += 0.030642807104826877;
                        } else {
                          result[0] += -0.07633959928192487;
                        }
                      } else {
                        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                          result[0] += 0.057930072725996797;
                        } else {
                          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.33373641967773615) ) ) {
                            result[0] += -0.03394137205152178;
                          } else {
                            result[0] += 0.07269607087016765;
                          }
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                        if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)144.5000000000000284) ) ) {
                          result[0] += -0.03206024882240121;
                        } else {
                          result[0] += 0.013114868256083927;
                        }
                      } else {
                        result[0] += -0.03829362550341885;
                      }
                    } else {
                      result[0] += -0.08064207920755287;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.318498134613038886) ) ) {
                      result[0] += 0.06611632035365006;
                    } else {
                      if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)144.5000000000000284) ) ) {
                        result[0] += 0.045972388956298814;
                      } else {
                        if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)147.5000000000000284) ) ) {
                          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.81940793991089045) ) ) {
                            if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.650573849678039995) ) ) {
                              result[0] += -0.2372034653670342;
                            } else {
                              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.672878742218019354) ) ) {
                                result[0] += 0.08118168447448434;
                              } else {
                                result[0] += -0.06895411671984168;
                              }
                            }
                          } else {
                            result[0] += 0.046604473964751425;
                          }
                        } else {
                          result[0] += 0.02111322247094952;
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.558241367340089667) ) ) {
                      if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)1.500000000000000222) ) ) {
                        if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)270.5000000000000568) ) ) {
                          result[0] += -0.051707982351964346;
                        } else {
                          result[0] += 0.009013511895275887;
                        }
                      } else {
                        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.873467922210695136) ) ) {
                          result[0] += -0.0024171320778513423;
                        } else {
                          result[0] += 0.05082762095651412;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                        if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                          result[0] += 0.0415549207115222;
                        } else {
                          if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                            result[0] += 0.038650717743420465;
                          } else {
                            if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
                              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)189.5000000000000284) ) ) {
                                if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                                  result[0] += 0.004887530518131723;
                                } else {
                                  result[0] += -0.1019578775463806;
                                }
                              } else {
                                result[0] += -0.051741855671959616;
                              }
                            } else {
                              result[0] += 0.0629864521765339;
                            }
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.397998809814454013) ) ) {
                          result[0] += -0.04385204930678758;
                        } else {
                          if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.597218394279480425) ) ) {
                            result[0] += -0.04395763223684285;
                          } else {
                            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.81940793991089045) ) ) {
                              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
                                result[0] += 0.014773603958214166;
                              } else {
                                result[0] += -0.02220245019545339;
                              }
                            } else {
                              result[0] += 0.04578766774775456;
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                  result[0] += 0.015172141337919224;
                } else {
                  result[0] += -0.10326857740758995;
                }
              } else {
                result[0] += 0.036610566868146704;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.873467922210695136) ) ) {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)144.5000000000000284) ) ) {
              result[0] += 0.04865856137281524;
            } else {
              result[0] += 0.0030688558910365407;
            }
          } else {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                  result[0] += 0.10641701549110323;
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.321723937988282138) ) ) {
                    result[0] += -0.07835658371703853;
                  } else {
                    result[0] += -0.01101393675619641;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.16594791412353693) ) ) {
                    result[0] += -0.05048258962613829;
                  } else {
                    result[0] += 0.01153080227649499;
                  }
                } else {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                    result[0] += 0.0022995091948789684;
                  } else {
                    if ( UNLIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)3.500000000000000444) ) ) {
                      result[0] += 0.08118302816774675;
                    } else {
                      result[0] += 0.006253130397195399;
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)189.5000000000000284) ) ) {
                  result[0] += 0.03288241186988955;
                } else {
                  result[0] += -0.083749991915756;
                }
              } else {
                result[0] += -0.07202739656608287;
              }
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
    result[0] += -0.0002600528246766778;
  } else {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.439558982849121982) ) ) {
      result[0] += -0.002194189197697354;
    } else {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.921060562133789951) ) ) {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.58713245391845881) ) ) {
            result[0] += 9.790501701759228e-05;
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.553655147552491123) ) ) {
              if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)108.5000000000000142) ) ) {
                result[0] += -0.04433078965074178;
              } else {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)111.5000000000000142) ) ) {
                  result[0] += 0.04930806233395729;
                } else {
                  result[0] += -0.005236516152973307;
                }
              }
            } else {
              result[0] += 0.011093353863399633;
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.558241367340089667) ) ) {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.344550132751465732) ) ) {
              if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += 0.043427619789114344;
              } else {
                result[0] += 0.005835930660127303;
              }
            } else {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)267.5000000000000568) ) ) {
                result[0] += -0.0036631660250226634;
              } else {
                result[0] += -0.039667188599376804;
              }
            }
          } else {
            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.745876312255860263) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.53326439857482999) ) ) {
                  result[0] += -0.03129650969104105;
                } else {
                  result[0] += 0.0033280153857659957;
                }
              } else {
                result[0] += -0.05778717845251772;
              }
            } else {
              if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += 0.07837382363466716;
              } else {
                result[0] += -0.05252900171307034;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.219419956207276279) ) ) {
                  if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.026367473822346346;
                  } else {
                    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.009395655376689212;
                    } else {
                      result[0] += 0.10597150710638388;
                    }
                  }
                } else {
                  result[0] += 0.01502759427857679;
                }
              } else {
                if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                  result[0] += 0.017695110213326266;
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.531673669815064365) ) ) {
                    result[0] += 0.0801208494232448;
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.102759599685669833) ) ) {
                      if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.329314231872559482) ) ) {
                          if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.198464870452881303) ) ) {
                            result[0] += 0.0029635377216785555;
                          } else {
                            result[0] += -0.044659581273422115;
                          }
                        } else {
                          result[0] += -0.055768587504442135;
                        }
                      } else {
                        if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                          result[0] += 0.01348612036172172;
                        } else {
                          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                            result[0] += -0.024721090259986728;
                          } else {
                            result[0] += -0.10551074041021609;
                          }
                        }
                      }
                    } else {
                      result[0] += 0.0028727511820025135;
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.129780292510988104) ) ) {
                  result[0] += -0.04374767199635418;
                } else {
                  result[0] += -0.007925742232159477;
                }
              } else {
                result[0] += 0.012980254175942785;
              }
            }
          } else {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.868834793567657693) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.318498134613038886) ) ) {
                  if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)300.5000000000000568) ) ) {
                    result[0] += -0.07797514629637706;
                  } else {
                    result[0] += 0.05829437903429389;
                  }
                } else {
                  if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.12938737869262873) ) ) {
                          result[0] += -0.014291348181518141;
                        } else {
                          result[0] += -0.09768236813193923;
                        }
                      } else {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.921100616455079013) ) ) {
                          result[0] += 0.010858664022722675;
                        } else {
                          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.617236852645874912) ) ) {
                            result[0] += -0.1214442654999213;
                          } else {
                            result[0] += 0.02099395181936438;
                          }
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.637949228286744052) ) ) {
                        result[0] += -0.01240541758891587;
                      } else {
                        result[0] += 0.02961128120453035;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.22008466720581232) ) ) {
                      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.921100616455079013) ) ) {
                        result[0] += -0.043644854238878594;
                      } else {
                        result[0] += 0.10130198089696517;
                      }
                    } else {
                      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.088880300521851474) ) ) {
                        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.637949228286744052) ) ) {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.99540567398071467) ) ) {
                            result[0] += 0.017126820253431613;
                          } else {
                            result[0] += 0.11775397221778904;
                          }
                        } else {
                          result[0] += -0.0942541532080422;
                        }
                      } else {
                        result[0] += 0.11423274328871275;
                      }
                    }
                  }
                }
              } else {
                result[0] += -0.13574107045004177;
              }
            } else {
              if ( UNLIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.497866153717041238) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.377930641174318183) ) ) {
                      result[0] += 0.05008724984754209;
                    } else {
                      result[0] += 0.00043474557366420045;
                    }
                  } else {
                    result[0] += -0.07122407794157859;
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.01520729064941584) ) ) {
                    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.006166534360223984;
                    } else {
                      if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += 0.046031621146997666;
                      } else {
                        if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.970085620880127397) ) ) {
                          result[0] += 0.006577350132097624;
                        } else {
                          result[0] += -0.17003809254358038;
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += 0.013543513838064323;
                    } else {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.921100616455079013) ) ) {
                        result[0] += 0.0688302948158731;
                      } else {
                        result[0] += 0.13753178590223672;
                      }
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
                  if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.01995429029868703;
                    } else {
                      result[0] += -0.009350538524290136;
                    }
                  } else {
                    result[0] += -0.0562121089659968;
                  }
                } else {
                  result[0] += 0.04627733712662452;
                }
              }
            }
          }
        } else {
          result[0] += -0.023316776966848574;
        }
      }
    }
  }
  if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.97070193290710538) ) ) {
      result[0] += -0.00012314303420899697;
    } else {
      if ( UNLIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)1.500000000000000222) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.474771499633789951) ) ) {
          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)312.5000000000000568) ) ) {
            result[0] += -0.08042158314600904;
          } else {
            result[0] += -0.014119999922430126;
          }
        } else {
          if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
            if ( LIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += -0.07089047276564463;
            } else {
              result[0] += -0.026238734258167137;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.744487762451173651) ) ) {
              if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)7.27828097343444913) ) ) {
                result[0] += -0.010200959583019906;
              } else {
                result[0] += -0.050643295233981644;
              }
            } else {
              if ( UNLIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += 0.010896564790953743;
              } else {
                result[0] += -0.03341371599104472;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
          if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
            if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)4.500000000000000888) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.58491539955139249) ) ) {
                result[0] += -0.06969894885754717;
              } else {
                if ( UNLIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)2.500000000000000444) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.213027238845826083) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.617236852645874912) ) ) {
                      result[0] += 0.05440280556740167;
                    } else {
                      result[0] += -0.03632907990203903;
                    }
                  } else {
                    if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)68.50000000000001421) ) ) {
                      result[0] += 0.0530166096253528;
                    } else {
                      result[0] += -0.013241027993691404;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.909102678298951083) ) ) {
                    if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                      result[0] += 0.03722400453906518;
                    } else {
                      if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.917405366897583452) ) ) {
                        if ( LIKELY( !(data[60].missing != -1) || (data[60].fvalue <= (double)6.000000000000000888) ) ) {
                          result[0] += 0.011954715463377728;
                        } else {
                          result[0] += 0.09559067753345181;
                        }
                      } else {
                        result[0] += 0.10667622397279247;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)68.50000000000001421) ) ) {
                      result[0] += 0.03833951915857904;
                    } else {
                      result[0] += -0.028066217356007662;
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)3.921924352645874468) ) ) {
                result[0] += 0.035950687911153946;
              } else {
                result[0] += -0.06377165388953021;
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.213027238845826083) ) ) {
              result[0] += 0.0246677709730851;
            } else {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += -0.024213278905417104;
              } else {
                result[0] += 0.02177933214831089;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.363266706466675693) ) ) {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += 0.004967537015679663;
                } else {
                  result[0] += -0.08138007971465351;
                }
              } else {
                result[0] += 0.06664021808739572;
              }
            } else {
              result[0] += -0.03851388068818756;
            }
          } else {
            result[0] += 0.013023312860266551;
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.449861526489258257) ) ) {
      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.673553824424744096) ) ) {
        result[0] += 0.012874699076005339;
      } else {
        result[0] += -0.0005261832116186556;
      }
    } else {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.53326439857482999) ) ) {
        if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.569433569908142534) ) ) {
            if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)296.5000000000000568) ) ) {
              result[0] += 0.010350139420233622;
            } else {
              result[0] += 0.05494454893725902;
            }
          } else {
            result[0] += 0.0013123763741764941;
          }
        } else {
          if ( UNLIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.89450073242187678) ) ) {
              if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)156.5000000000000284) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.58713245391845881) ) ) {
                  if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += -0.1436676382770608;
                  } else {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.835998296737671787) ) ) {
                      result[0] += -0.08176545164418228;
                    } else {
                      result[0] += 0.006664717832888466;
                    }
                  }
                } else {
                  result[0] += 0.005572200215027408;
                }
              } else {
                result[0] += -0.001829997221905293;
              }
            } else {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
                  result[0] += 0.015367594760488415;
                } else {
                  result[0] += 0.06324179101695715;
                }
              } else {
                result[0] += -0.01478269327483174;
              }
            }
          } else {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.861792564392090288) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.510617971420288974) ) ) {
                result[0] += 0.010839115478923394;
              } else {
                if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.051747083663941318) ) ) {
                  result[0] += -0.00703461080759208;
                } else {
                  result[0] += -0.06171537495460801;
                }
              }
            } else {
              result[0] += -0.04514053093747464;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)192.5000000000000284) ) ) {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.249904870986938921) ) ) {
              if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)2.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.067782521247864214) ) ) {
                  result[0] += -0.09418200498524548;
                } else {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.022341651432185288;
                  } else {
                    if ( UNLIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.021854963189360567;
                    } else {
                      result[0] += 0.004619752637423847;
                    }
                  }
                }
              } else {
                result[0] += 0.02270693416212389;
              }
            } else {
              result[0] += 0.017759270682242357;
            }
          } else {
            if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)2.500000000000000444) ) ) {
              result[0] += 0.007011441262836704;
            } else {
              if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.863673448562622958) ) ) {
                result[0] += -0.005988041430589048;
              } else {
                result[0] += -0.04859845204203934;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
              if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.321723937988282138) ) ) {
                  result[0] += -0.05002246034988488;
                } else {
                  result[0] += 0.09234513227083639;
                }
              } else {
                result[0] += -0.08196318136369124;
              }
            } else {
              if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                result[0] += 0.059647806893299585;
              } else {
                result[0] += 0.01085057149382073;
              }
            }
          } else {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += 0.014275602647493627;
            } else {
              result[0] += -0.06507713089600277;
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
    result[0] += -0.0002483405499496936;
  } else {
    if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.449861526489258257) ) ) {
      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.436733961105347568) ) ) {
        result[0] += -0.0012395148384726294;
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
          result[0] += -0.0037441840052350748;
        } else {
          result[0] += 0.00513453029735007;
        }
      }
    } else {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.53326439857482999) ) ) {
        if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
          result[0] += 0.0026102418410861913;
        } else {
          if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.03089046478271662) ) ) {
              if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.375260829925538886) ) ) {
                  result[0] += -0.061407318357090845;
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.835998296737671787) ) ) {
                    result[0] += 0.08296468014608495;
                  } else {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.917405366897583452) ) ) {
                      result[0] += 0.04441230100225714;
                    } else {
                      if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                        result[0] += -0.09639134904517882;
                      } else {
                        result[0] += 0.02046182604447717;
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                  if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)300.5000000000000568) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.704609394073488104) ) ) {
                      result[0] += 0.0350439716178064;
                    } else {
                      result[0] += -0.03750829965939262;
                    }
                  } else {
                    result[0] += -0.10796111542802958;
                  }
                } else {
                  if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.835998296737671787) ) ) {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.524927973747253862) ) ) {
                        result[0] += -0.07347499518028132;
                      } else {
                        result[0] += 0.05046452175368894;
                      }
                    } else {
                      result[0] += 0.08939848553084133;
                    }
                  } else {
                    result[0] += -0.013316933780886895;
                  }
                }
              }
            } else {
              result[0] += 0.02333346138445002;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.9353518486022967) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.510617971420288974) ) ) {
                result[0] += 0.015289125404873797;
              } else {
                result[0] += -0.027538494191401832;
              }
            } else {
              result[0] += -0.03933354782166695;
            }
          }
        }
      } else {
        if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.241523027420044833) ) ) {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)144.5000000000000284) ) ) {
              result[0] += -0.04446463895374476;
            } else {
              result[0] += 0.0024977388871156225;
            }
          } else {
            if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                  if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                    if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.138333082199097124) ) ) {
                        if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                          result[0] += -0.0055835495829801894;
                        } else {
                          result[0] += -0.08109781729098531;
                        }
                      } else {
                        result[0] += 0.00902653238687673;
                      }
                    } else {
                      result[0] += 0.029775450050023286;
                    }
                  } else {
                    result[0] += 0.06675820347188499;
                  }
                } else {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
                    if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += 0.05102341766282659;
                    } else {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.715336322784424716) ) ) {
                        result[0] += -0.053181836825264575;
                      } else {
                        result[0] += 0.03507970775513639;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.120439291000367099) ) ) {
                      result[0] += -0.13827022631729433;
                    } else {
                      result[0] += 0.02506930453499708;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                  result[0] += 0.01032248804388418;
                } else {
                  if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += -0.041365083841440345;
                  } else {
                    result[0] += 0.016318703784838833;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.597218394279480425) ) ) {
                  result[0] += -0.034905248042251244;
                } else {
                  result[0] += 0.0831246450630473;
                }
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.7592954635620135) ) ) {
                  if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                    if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.04314378702741904;
                    } else {
                      result[0] += -0.011323095793491353;
                    }
                  } else {
                    if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.597218394279480425) ) ) {
                      result[0] += 0.01985631968935671;
                    } else {
                      result[0] += -0.06484983391189822;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.855921268463135654) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.04690074920654475) ) ) {
                      result[0] += -0.01123782942771017;
                    } else {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.088880300521851474) ) ) {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.637949228286744052) ) ) {
                          result[0] += 0.052479856525095715;
                        } else {
                          result[0] += -0.08295005026819395;
                        }
                      } else {
                        result[0] += 0.16207710944461282;
                      }
                    }
                  } else {
                    result[0] += 0.10157458103030781;
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.780892848968506748) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.764287948608400214) ) ) {
                  result[0] += 0.02733048594322819;
                } else {
                  if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)48.00000000000000711) ) ) {
                    if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                      result[0] += 0.0021237640670345973;
                    } else {
                      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                        result[0] += -0.03239206534201946;
                      } else {
                        result[0] += -0.10047453752636003;
                      }
                    }
                  } else {
                    result[0] += 0.024096018359095788;
                  }
                }
              } else {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                  result[0] += 0.01625396126442767;
                } else {
                  result[0] += -0.0731755636705218;
                }
              }
            } else {
              if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.989220380783081943) ) ) {
                  result[0] += 0.04779912762761323;
                } else {
                  result[0] += -0.004986904103120665;
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.704609394073488104) ) ) {
                  result[0] += 0.012199906829500683;
                } else {
                  result[0] += 0.03187811873600709;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.007332723471072515;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.22008466720581232) ) ) {
                  result[0] += 0.0007589052073310588;
                } else {
                  result[0] += -0.028469010797362505;
                }
              }
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += 0.01450908007549396;
              } else {
                result[0] += -0.061542184769782064;
              }
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
    if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.79084348678589045) ) ) {
        if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.173316955566407138) ) ) {
          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.321723937988282138) ) ) {
                result[0] += 0.0005185806686413019;
              } else {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.055782999798725225;
                } else {
                  result[0] += 0.004418055412587893;
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.474771499633789951) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.450390577316285068) ) ) {
                  if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += -0.032478229474121524;
                    } else {
                      result[0] += 0.0396414322641494;
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.556798219680787021) ) ) {
                      result[0] += 0.02695126814897468;
                    } else {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.029068946838379794) ) ) {
                        result[0] += 0.017295525727183955;
                      } else {
                        result[0] += -0.017211995389898187;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.986973047256470615) ) ) {
                    result[0] += 0.009506018532012675;
                  } else {
                    result[0] += -0.05513709539166835;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                  result[0] += 0.008963611542418736;
                } else {
                  result[0] += -0.04563873633592225;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)66.50000000000001421) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.921100616455079013) ) ) {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.617236852645874912) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
                      result[0] += -0.045922933986987276;
                    } else {
                      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.621853828430177558) ) ) {
                          result[0] += -0.048016062828640405;
                        } else {
                          result[0] += 0.011144112274620536;
                        }
                      } else {
                        result[0] += 0.02359213267629592;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.020127415657043901) ) ) {
                      result[0] += -0.0445911721685645;
                    } else {
                      result[0] += 0.07545960269049425;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.842459201812745917) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.637949228286744052) ) ) {
                      result[0] += 0.005045125454882261;
                    } else {
                      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                        result[0] += -0.012560515474878978;
                      } else {
                        result[0] += -0.06887283645855072;
                      }
                    }
                  } else {
                    result[0] += -0.05855867074029424;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.773543357849121982) ) ) {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                    result[0] += -0.12912958462920754;
                  } else {
                    result[0] += 0.009510878851609788;
                  }
                } else {
                  result[0] += 0.030076354916917014;
                }
              }
            } else {
              result[0] += 0.009717613979702444;
            }
          }
        } else {
          result[0] += -0.026399140333171114;
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.262283086776734287) ) ) {
          if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.605039834976196733) ) ) {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += 0.028338671391543307;
              } else {
                if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += 0.06288699158941972;
                } else {
                  result[0] += -0.03677311804194599;
                }
              }
            } else {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += -0.017913792504266907;
              } else {
                result[0] += 0.02298182165308164;
              }
            }
          } else {
            result[0] += -0.004911346201221253;
          }
        } else {
          if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
              if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)65.50000000000001421) ) ) {
                result[0] += 0.03638177869500722;
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.029068946838379794) ) ) {
                  if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.029921936886459107;
                  } else {
                    result[0] += -0.01623552841011182;
                  }
                } else {
                  if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.03611005234000259;
                  } else {
                    result[0] += 0.013008790157408932;
                  }
                }
              }
            } else {
              result[0] += -0.024306545012935214;
            }
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
              result[0] += -0.027532781303927407;
            } else {
              result[0] += -0.06455238451104546;
            }
          }
        }
      }
    } else {
      result[0] += -0.0009700432405928727;
    }
  } else {
    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.182021141052246982) ) ) {
      result[0] += 0.00031605159470899074;
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
        if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
          if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.861792564392090288) ) ) {
            if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += 0.0068766366504619;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.773543357849121982) ) ) {
                result[0] += -0.06610095047044941;
              } else {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.023886388790196184;
                } else {
                  if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.350240230560303178) ) ) {
                    result[0] += 0.028842447160862785;
                  } else {
                    result[0] += -0.035959272303517774;
                  }
                }
              }
            }
          } else {
            result[0] += 0.013752653327775652;
          }
        } else {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.342454433441162998) ) ) {
            result[0] += -0.046888017063002455;
          } else {
            result[0] += -0.014473573750348447;
          }
        }
      } else {
        if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
          result[0] += -0.0015151207322916804;
        } else {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.029068946838379794) ) ) {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.219419956207276279) ) ) {
              result[0] += 0.019992553656924063;
            } else {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.02595988539717405;
                } else {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.855921268463135654) ) ) {
                    result[0] += 0.042700354831121404;
                  } else {
                    result[0] += -0.006282300116822123;
                  }
                }
              } else {
                result[0] += -0.02686242879284875;
              }
            }
          } else {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)294.5000000000000568) ) ) {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.761470437049866167) ) ) {
                  result[0] += 0.020374087770206606;
                } else {
                  result[0] += -0.11783569613928041;
                }
              } else {
                result[0] += -0.006006848036066215;
              }
            } else {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.909855604171753818) ) ) {
                  result[0] += -0.033449435633495526;
                } else {
                  result[0] += 0.0035452593712789818;
                }
              } else {
                result[0] += 0.06869204485608453;
              }
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[60].missing != -1) || (data[60].fvalue <= (double)1.00000001800250948e-35) ) ) {
    result[0] += 0.09253863305628762;
  } else {
    if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
      if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.028861761093140537) ) ) {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
              result[0] += -0.01886715614550475;
            } else {
              if ( UNLIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.47712564468383967) ) ) {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.142630577087403232) ) ) {
                    result[0] += 0.003082764812691585;
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.189540147781372958) ) ) {
                      result[0] += 0.03805696448324524;
                    } else {
                      result[0] += 0.00959945874459456;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                    result[0] += 0.00471047766500798;
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.431901693344116655) ) ) {
                      result[0] += -0.0133734917026994;
                    } else {
                      result[0] += -0.06922559201318956;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.216319084167481357) ) ) {
                  if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += 0.0035939934328792587;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.7592954635620135) ) ) {
                      result[0] += -0.010450748137031108;
                    } else {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.531673669815064365) ) ) {
                        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.094205617904663974) ) ) {
                            result[0] += 0.020149112775965903;
                          } else {
                            result[0] += -0.019977361830510357;
                          }
                        } else {
                          result[0] += -0.04317004275265148;
                        }
                      } else {
                        if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)107.5000000000000142) ) ) {
                          result[0] += 0.029258590877819565;
                        } else {
                          result[0] += -0.001033171181408834;
                        }
                      }
                    }
                  }
                } else {
                  result[0] += 0.007986561789646313;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
              result[0] += -0.04233598579797747;
            } else {
              result[0] += 0.011199453901758478;
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.828941345214844638) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.807895898818970615) ) ) {
              if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
                if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                  result[0] += 0.06511241793168669;
                } else {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += -0.0006511104022550212;
                  } else {
                    result[0] += 0.048018382846671696;
                  }
                }
              } else {
                result[0] += 0.0010563887478812102;
              }
            } else {
              result[0] += -0.014745197606385669;
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.433569431304932529) ) ) {
              if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.015136069899729624;
              } else {
                result[0] += -0.017472308709710067;
              }
            } else {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.932935476303101474) ) ) {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.53326439857482999) ) ) {
                      result[0] += -0.040526409921345846;
                    } else {
                      result[0] += 0.014070611096090256;
                    }
                  } else {
                    result[0] += -0.06984914981356853;
                  }
                } else {
                  result[0] += -0.0701975549169979;
                }
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.921100616455079013) ) ) {
                  result[0] += -0.02296453463176442;
                } else {
                  result[0] += 0.0647287874139101;
                }
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.142630577087403232) ) ) {
            result[0] += -0.0015466950252428485;
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.587308406829834873) ) ) {
                  result[0] += -0.057761842137811326;
                } else {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.701225757598877397) ) ) {
                      result[0] += -0.031021666309852765;
                    } else {
                      result[0] += -0.0596871753095683;
                    }
                  } else {
                    if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += -0.034970103390383;
                    } else {
                      result[0] += 0.002094153609026141;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.497866153717041238) ) ) {
                  result[0] += 0.016756258481884483;
                } else {
                  result[0] += -0.03490936084187008;
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.590985536575318271) ) ) {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.56941866874694913) ) ) {
                  result[0] += -0.027455326471045694;
                } else {
                  result[0] += -0.08908423426084958;
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.142630577087403232) ) ) {
                  result[0] += 0.03641462360223774;
                } else {
                  result[0] += -0.00027128964318280793;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.686429500579835761) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.778982400894165927) ) ) {
              result[0] += 0.04244338668128407;
            } else {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)46.50000000000000711) ) ) {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.56941866874694913) ) ) {
                    if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.078787803649903232) ) ) {
                      result[0] += 0.013867512080577911;
                    } else {
                      result[0] += -0.052856280201117205;
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.828941345214844638) ) ) {
                      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
                        result[0] += -0.09116608760994524;
                      } else {
                        result[0] += 0.01413605049362213;
                      }
                    } else {
                      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)5.500000000000000888) ) ) {
                        if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
                          result[0] += 0.033261451482741004;
                        } else {
                          if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.433743238449097568) ) ) {
                            result[0] += 0.01896774934489402;
                          } else {
                            result[0] += 0.0952787580364371;
                          }
                        }
                      } else {
                        result[0] += 0.005677139242355545;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.986973047256470615) ) ) {
                    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
                      result[0] += -0.012583725464826305;
                    } else {
                      if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)275.5000000000000568) ) ) {
                        result[0] += 0.04158658881285185;
                      } else {
                        if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)0.8958797454833985485) ) ) {
                          result[0] += -0.060456478238822935;
                        } else {
                          result[0] += 0.0863710602304224;
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
                      result[0] += -0.012532540109201845;
                    } else {
                      result[0] += 0.022383279991641886;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.932935476303101474) ) ) {
                    result[0] += 0.026520255095007567;
                  } else {
                    result[0] += -0.04944712614187957;
                  }
                } else {
                  result[0] += -0.05086309032699896;
                }
              }
            }
          } else {
            result[0] += -0.027251558261281006;
          }
        }
      }
    } else {
      result[0] += 0.0004871484580947615;
    }
  }
  if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.579273939132691318) ) ) {
      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.088880300521851474) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.873467922210695136) ) ) {
            result[0] += -0.0061031062655965375;
          } else {
            result[0] += 0.002507721169179189;
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.158010244369507724) ) ) {
            result[0] += -0.002835017163811295;
          } else {
            result[0] += 0.010666766086741143;
          }
        }
      } else {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.637949228286744052) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.54081821441650568) ) ) {
            result[0] += 0.006430420067287707;
          } else {
            if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.497866153717041238) ) ) {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                  result[0] += -0.04926699162305797;
                } else {
                  result[0] += 0.005230724270137624;
                }
              } else {
                result[0] += -0.071812084049413;
              }
            } else {
              result[0] += 0.008758545990326605;
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.932935476303101474) ) ) {
            result[0] += -4.2969241033534055e-05;
          } else {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
              result[0] += -0.0350569090157305;
            } else {
              result[0] += -0.06967375855517653;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.040716171264650214) ) ) {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.623839378356934482) ) ) {
              result[0] += -0.07833700188675398;
            } else {
              result[0] += -0.02612293591526731;
            }
          } else {
            if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)89.50000000000001421) ) ) {
              if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += -0.03458876151210729;
              } else {
                result[0] += -0.0024979558994612087;
              }
            } else {
              result[0] += -0.040137593868315974;
            }
          }
        } else {
          result[0] += 0.0008862830480561505;
        }
      } else {
        result[0] += 0.0026317734246964343;
      }
    }
  } else {
    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.182021141052246982) ) ) {
      if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.497866153717041238) ) ) {
        if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)100.5000000000000142) ) ) {
          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
            result[0] += -0.006269656840839766;
          } else {
            result[0] += -0.036544447959901735;
          }
        } else {
          if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.853637218475342685) ) ) {
            result[0] += -6.906738554497655e-05;
          } else {
            if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.27947616577148615) ) ) {
                  result[0] += 0.02255937619587729;
                } else {
                  result[0] += -0.0733434860525858;
                }
              } else {
                result[0] += 0.03391087651868332;
              }
            } else {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)197.5000000000000284) ) ) {
                if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.01638439604999045;
                } else {
                  result[0] += -0.036166419953921995;
                }
              } else {
                if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                  if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)238.5000000000000284) ) ) {
                    result[0] += -0.06416294422022105;
                  } else {
                    result[0] += 0.018751500279317282;
                  }
                } else {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                    result[0] += -0.024358911886377326;
                  } else {
                    result[0] += -0.09667127722226865;
                  }
                }
              }
            }
          }
        }
      } else {
        result[0] += 0.01321922506072315;
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
        if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
          if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.861792564392090288) ) ) {
            if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += 0.005362188644798685;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.009402275085450107) ) ) {
                result[0] += -0.0850880382099922;
              } else {
                result[0] += -0.012846075245295819;
              }
            }
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += 0.015622632606822145;
            } else {
              result[0] += -0.011938490543298859;
            }
          }
        } else {
          result[0] += -0.02293500070245755;
        }
      } else {
        if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.029068946838379794) ) ) {
            result[0] += 0.009632984159797128;
          } else {
            result[0] += -0.005503895809376649;
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.029068946838379794) ) ) {
            if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.407877445220948154) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.12938737869262873) ) ) {
                    result[0] += 0.05565047260870412;
                  } else {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.325443029403687412) ) ) {
                      result[0] += -0.022636053624738944;
                    } else {
                      result[0] += 0.034934200428732846;
                    }
                  }
                } else {
                  result[0] += -0.045773255427486274;
                }
              } else {
                result[0] += -0.02601562469718669;
              }
            } else {
              result[0] += -0.007768474507040674;
            }
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
              if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.397998809814454013) ) ) {
                  if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.219419956207276279) ) ) {
                      result[0] += -0.05960075880098958;
                    } else {
                      if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                        result[0] += -0.040243215505352564;
                      } else {
                        result[0] += 0.019929210629359623;
                      }
                    }
                  } else {
                    result[0] += 0.08558300748932132;
                  }
                } else {
                  result[0] += 0.01299787401187679;
                }
              } else {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
                  if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)266.5000000000000568) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.467917680740357333) ) ) {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.51517200469970881) ) ) {
                        result[0] += 0.03669367643095626;
                      } else {
                        result[0] += 0.1455279752272761;
                      }
                    } else {
                      result[0] += 0.03657989960430934;
                    }
                  } else {
                    result[0] += -0.01077408857074871;
                  }
                } else {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                    result[0] += -0.08583478734725665;
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.088880300521851474) ) ) {
                      result[0] += -0.00819501499346156;
                    } else {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.96495962142944514) ) ) {
                        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                          result[0] += -0.00026710732829426763;
                        } else {
                          result[0] += 0.04241111448571343;
                        }
                      } else {
                        result[0] += 0.06923064832272172;
                      }
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.761470437049866167) ) ) {
                if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.008180452431997656;
                } else {
                  result[0] += -0.08656412432009043;
                }
              } else {
                result[0] += 0.08800069942528309;
              }
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
    if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.938058137893677646) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.105651378631592685) ) ) {
          if ( LIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.962127923965454546) ) ) {
                result[0] += 0.0049741544781966365;
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.431901693344116655) ) ) {
                  if ( UNLIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)24.00000000000000355) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
                      result[0] += -0.051944447508966686;
                    } else {
                      result[0] += 0.029177175020713375;
                    }
                  } else {
                    result[0] += -0.022910216249892392;
                  }
                } else {
                  if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += -0.02727217386190461;
                  } else {
                    result[0] += -0.11229852383242994;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.014808795714916199;
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.553655147552491123) ) ) {
                  result[0] += -0.01758642223916113;
                } else {
                  result[0] += 0.006898952964127321;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
              if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)9.017723560333253729) ) ) {
                if ( LIKELY( !(data[56].missing != -1) || (data[56].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += -0.0006026017852495359;
                } else {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.166635274887085849) ) ) {
                    result[0] += -0.019322595308967905;
                  } else {
                    result[0] += 0.011263837704515327;
                  }
                }
              } else {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                  result[0] += 0.011490459986776683;
                } else {
                  result[0] += -0.04823243785900286;
                }
              }
            } else {
              result[0] += 0.02907142458479401;
            }
          }
        } else {
          if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.523457050323487216) ) ) {
            if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.497866153717041238) ) ) {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
                result[0] += 0.0024269472969706118;
              } else {
                if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)7.208230257034302646) ) ) {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.214365959167481357) ) ) {
                      result[0] += 0.003916295809214994;
                    } else {
                      result[0] += -0.00914658200834351;
                    }
                  } else {
                    if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1.497866153717041238) ) ) {
                      result[0] += 0.012852356168922005;
                    } else {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.909855604171753818) ) ) {
                        result[0] += -0.044135900920005154;
                      } else {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.158952236175537998) ) ) {
                          result[0] += 0.0057981104500929165;
                        } else {
                          result[0] += -0.02414069356620766;
                        }
                      }
                    }
                  }
                } else {
                  result[0] += -0.02289268297943597;
                }
              }
            } else {
              result[0] += -0.04040329671354569;
            }
          } else {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.019634111263153748;
            } else {
              result[0] += -0.06951968725596443;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.553712725639343706) ) ) {
          result[0] += 0.008852285711688451;
        } else {
          if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += -0.03949248503102289;
          } else {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.029068946838379794) ) ) {
                  result[0] += 0.012597135170866662;
                } else {
                  result[0] += -0.00878123774677486;
                }
              } else {
                result[0] += -0.05965248269816025;
              }
            } else {
              if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1.497866153717041238) ) ) {
                result[0] += 0.019791947854678797;
              } else {
                result[0] += -0.021824203496477738;
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
        result[0] += -0.002737099601987467;
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.267844915390015537) ) ) {
          if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
            result[0] += 0.0020867166973981646;
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.262283086776734287) ) ) {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.714014530181885654) ) ) {
                result[0] += -0.01633848858092658;
              } else {
                result[0] += -0.04793597020587703;
              }
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.029068946838379794) ) ) {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.855921268463135654) ) ) {
                    result[0] += 0.00048683252525674026;
                  } else {
                    result[0] += 0.027772424043995422;
                  }
                } else {
                  result[0] += -0.08201466708571008;
                }
              } else {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.481121778488159624) ) ) {
                  result[0] += -0.007450986531423346;
                } else {
                  if ( UNLIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
                    if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                      result[0] += -0.027425074040236714;
                    } else {
                      result[0] += -0.1083291662425334;
                    }
                  } else {
                    result[0] += -0.027229346157556063;
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)9.393244743347169745) ) ) {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.24492526054382413) ) ) {
                if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)4.620046615600586826) ) ) {
                  result[0] += -0.006873883466415308;
                } else {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.021152361376012434;
                  } else {
                    result[0] += 0.004572579070720658;
                  }
                }
              } else {
                if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += 0.03601901858337753;
                } else {
                  result[0] += 0.0013428527234265897;
                }
              }
            } else {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += 0.008369813426880214;
              } else {
                result[0] += 0.09332706414289131;
              }
            }
          } else {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.020151551018388184;
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.28299736976623624) ) ) {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.051747083663941318) ) ) {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                    result[0] += 0.010158880532025185;
                  } else {
                    result[0] += -0.04336795506712316;
                  }
                } else {
                  if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.012675821781158891) ) ) {
                    result[0] += 0.012735242353119845;
                  } else {
                    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.01854798612549769;
                    } else {
                      result[0] += -0.06327843355655603;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.001854659707636608;
                } else {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.051747083663941318) ) ) {
                    result[0] += 0.006278207195192984;
                  } else {
                    result[0] += 0.024280498582133825;
                  }
                }
              }
            }
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.182021141052246982) ) ) {
      if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.497866153717041238) ) ) {
        result[0] += 4.372373733879304e-05;
      } else {
        result[0] += 0.012695925415420651;
      }
    } else {
      result[0] += 0.004750471933184023;
    }
  }
  if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
    if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)6144.000000000000909) ) ) {
      if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)3.500000000000000444) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.028861761093140537) ) ) {
            if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.620046615600586826) ) ) {
              if ( UNLIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.051912069320679599) ) ) {
                    if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.002105005037031493;
                    } else {
                      result[0] += 0.023624142050934868;
                    }
                  } else {
                    result[0] += -0.04879451208027519;
                  }
                } else {
                  if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.388237953186036044) ) ) {
                    if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += 0.013173827929565727;
                    } else {
                      if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.384246587753296343) ) ) {
                        result[0] += 0.02250457091789046;
                      } else {
                        result[0] += 0.08619641184078025;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.510617971420288974) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.1746091842651385) ) ) {
                        result[0] += 0.1012980922051207;
                      } else {
                        result[0] += -0.0028748889236852146;
                      }
                    } else {
                      result[0] += -0.04539631288237712;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.216319084167481357) ) ) {
                  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.009402275085450107) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += 0.14764151596645994;
                      } else {
                        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.932935476303101474) ) ) {
                          if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.166635274887085849) ) ) {
                            result[0] += 0.010969082412051464;
                          } else {
                            result[0] += 0.08860878316919531;
                          }
                        } else {
                          result[0] += 0.11474620473744582;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)1.242453336715698464) ) ) {
                        result[0] += -0.06534233056883025;
                      } else {
                        result[0] += 0.00019416285970904628;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += -0.03815827886163794;
                    } else {
                      result[0] += -0.005964461376941759;
                    }
                  }
                } else {
                  result[0] += 0.008665257593645184;
                }
              }
            } else {
              if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += -0.015508522684717314;
              } else {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)36.50000000000000711) ) ) {
                  result[0] += 0.014558210400595645;
                } else {
                  result[0] += -0.0036235692227475132;
                }
              }
            }
          } else {
            result[0] += -0.025279398497627177;
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.272946834564209873) ) ) {
            if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
              result[0] += 0.00326796383808851;
            } else {
              result[0] += 0.03594219941827494;
            }
          } else {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.56941866874694913) ) ) {
              if ( UNLIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)4.500000000000000888) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.743881702423096591) ) ) {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
                    if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.241523027420044833) ) ) {
                        result[0] += -0.049901370958571875;
                      } else {
                        result[0] += -0.01267486747697156;
                      }
                    } else {
                      if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
                        result[0] += 0.03175315890513794;
                      } else {
                        result[0] += -0.014375367987398458;
                      }
                    }
                  } else {
                    result[0] += 0.022060541645991125;
                  }
                } else {
                  result[0] += 0.033476420866322576;
                }
              } else {
                if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.012675821781158891) ) ) {
                  result[0] += 0.021836231021383642;
                } else {
                  result[0] += -0.03702530293866485;
                }
              }
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.617236852645874912) ) ) {
                if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += -0.054787586962222905;
                } else {
                  result[0] += 0.0759701978534921;
                }
              } else {
                result[0] += 0.04765629823197853;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)3.500000000000000444) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.241523027420044833) ) ) {
            if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.413873195648194248) ) ) {
              result[0] += -0.0017340958991680418;
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.737386107444763628) ) ) {
                result[0] += -0.003793204522856074;
              } else {
                result[0] += -0.030780764943476764;
              }
            }
          } else {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.264029741287232333) ) ) {
                if ( UNLIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.008307943318886261;
                } else {
                  if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)167.5000000000000284) ) ) {
                    result[0] += -0.0616394812080437;
                  } else {
                    result[0] += -0.018906724439869668;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.764287948608400214) ) ) {
                    result[0] += -0.07498146440325751;
                  } else {
                    result[0] += -0.03168906728294953;
                  }
                } else {
                  if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += -0.05478132627951731;
                  } else {
                    if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += -0.032955025930636504;
                    } else {
                      result[0] += 0.010477480974296538;
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.617236852645874912) ) ) {
                result[0] += 0.0010266764817222997;
              } else {
                if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.717199802398682529) ) ) {
                  if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.676220536231995073) ) ) {
                    result[0] += 0.11765608287040615;
                  } else {
                    result[0] += 0.7977211873039782;
                  }
                } else {
                  result[0] += -0.055060477080856624;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.686429500579835761) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.778982400894165927) ) ) {
              result[0] += 0.03699226592636817;
            } else {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)46.50000000000000711) ) ) {
                  result[0] += 0.01592199201038428;
                } else {
                  if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.951942920684815341) ) ) {
                    result[0] += 0.0068353429451278695;
                  } else {
                    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                      result[0] += -0.008810313847119786;
                    } else {
                      result[0] += -0.054934439969593796;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)4.500000000000000888) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.932935476303101474) ) ) {
                    result[0] += 0.023752633724853155;
                  } else {
                    result[0] += -0.04468900196623404;
                  }
                } else {
                  if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)4.973273515701294833) ) ) {
                    result[0] += 0.009498347590629858;
                  } else {
                    result[0] += -0.06326538294334165;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)5.500000000000000888) ) ) {
              result[0] += -0.01889148243709525;
            } else {
              result[0] += -0.06869634757850067;
            }
          }
        }
      }
    } else {
      result[0] += -0.02002053436506747;
    }
  } else {
    result[0] += 0.0005133764636484692;
  }
  if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
    if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)6144.000000000000909) ) ) {
      result[0] += 0.00023004069349852495;
    } else {
      if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
        result[0] += -0.09465146690095538;
      } else {
        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
          result[0] += -0.10448097232512273;
        } else {
          if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)65.50000000000001421) ) ) {
            result[0] += 0.0483813340051137;
          } else {
            result[0] += 0.012879729996456153;
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)6144.000000000000909) ) ) {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.637949228286744052) ) ) {
            if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += 0.0017555201010545274;
                } else {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.017069212969683296;
                  } else {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.166635274887085849) ) ) {
                      result[0] += -0.013576487265829107;
                    } else {
                      result[0] += 0.09335352289944404;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += -0.05615165546555567;
                } else {
                  if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)24.00000000000000355) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.242453336715698464) ) ) {
                      result[0] += -0.16110919904803467;
                    } else {
                      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.481121778488159624) ) ) {
                        result[0] += -0.05413053431536435;
                      } else {
                        if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.868834793567657693) ) ) {
                          result[0] += -0.009660582061811794;
                        } else {
                          result[0] += 0.15403804358742318;
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)1.497866153717041238) ) ) {
                      result[0] += 0.09506788269743004;
                    } else {
                      result[0] += -0.023634148314004094;
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += 0.014166677428431505;
              } else {
                result[0] += -0.005314935405944208;
              }
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.028861761093140537) ) ) {
              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += -0.02558157451681364;
              } else {
                result[0] += -0.08736619654073675;
              }
            } else {
              result[0] += -0.0883957926584491;
            }
          }
        } else {
          if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
            if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.987706661224366123) ) ) {
              result[0] += 0.04760097424385677;
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.216319084167481357) ) ) {
                result[0] += 0.04147284915716263;
              } else {
                result[0] += -0.053865152860445215;
              }
            }
          } else {
            if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)3.134548187255859819) ) ) {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.53326439857482999) ) ) {
                  if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)243.5000000000000284) ) ) {
                    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.012437530528361232;
                    } else {
                      result[0] += -0.041766446203173595;
                    }
                  } else {
                    result[0] += -0.038319627437610304;
                  }
                } else {
                  result[0] += -0.03436109227001274;
                }
              } else {
                if ( LIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)12.00000000000000178) ) ) {
                  result[0] += -0.004371035563925314;
                } else {
                  result[0] += 0.03277293413201717;
                }
              }
            } else {
              result[0] += 0.005128949417623708;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
            if ( UNLIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.553712725639343706) ) ) {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.43267917633056818) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += 0.049336705721711026;
                    } else {
                      if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                        result[0] += 0.008364603407607862;
                      } else {
                        result[0] += -0.05270418558015273;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
                      result[0] += 0.02974474796705623;
                    } else {
                      if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += 0.009854975739555447;
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.47712564468383967) ) ) {
                          result[0] += 0.06899755018884811;
                        } else {
                          if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)2.44140100479126021) ) ) {
                            result[0] += -0.10758244954459756;
                          } else {
                            result[0] += 0.028996364200005755;
                          }
                        }
                      }
                    }
                  }
                } else {
                  result[0] += 0.031825159333554454;
                }
              } else {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                  if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)43.50000000000000711) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                      result[0] += 0.039711532054866557;
                    } else {
                      result[0] += -0.05468656418177962;
                    }
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.105651378631592685) ) ) {
                      result[0] += 0.003848680760426216;
                    } else {
                      result[0] += -0.01896833905846146;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.439304351806642401) ) ) {
                      result[0] += -0.03228957259713557;
                    } else {
                      result[0] += -0.07005347633493951;
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.58713245391845881) ) ) {
                      result[0] += 0.0038496395805066724;
                    } else {
                      result[0] += -0.04946205823432413;
                    }
                  }
                }
              }
            } else {
              result[0] += -0.0017116967653304739;
            }
          } else {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.552201986312867099) ) ) {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.780738830566407138) ) ) {
                result[0] += -0.0011542502073878314;
              } else {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.020657935076734957;
                } else {
                  result[0] += -0.0019962050994896787;
                }
              }
            } else {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.054550735016320996;
              } else {
                result[0] += -0.0009003341726783273;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)2.500000000000000444) ) ) {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
              result[0] += -0.005785446318953978;
            } else {
              result[0] += -0.039527346288159924;
            }
          } else {
            result[0] += -0.058760831733240954;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.377930641174318183) ) ) {
        if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)6.000000000000000888) ) ) {
          result[0] += -0.024875846399339565;
        } else {
          result[0] += -0.07234271343125369;
        }
      } else {
        if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.662244915962219682) ) ) {
            result[0] += 0.045435104920574926;
          } else {
            result[0] += -0.0780358019366089;
          }
        } else {
          if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
            result[0] += 0.008395873589353692;
          } else {
            result[0] += -0.06946213099421111;
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.00000001800250948e-35) ) ) {
    result[0] += 0.08482304375406559;
  } else {
    if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)25.50000000000000355) ) ) {
      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.938033580780031073) ) ) {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.285887241363526279) ) ) {
            if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.556798219680787021) ) ) {
                  result[0] += -0.026471981949778612;
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.825422286987305576) ) ) {
                    result[0] += -0.0017696645354090227;
                  } else {
                    result[0] += 0.014475436191091927;
                  }
                }
              } else {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.53326439857482999) ) ) {
                    result[0] += -0.11648638693197591;
                  } else {
                    result[0] += -0.006559054568006816;
                  }
                } else {
                  result[0] += -0.07164476734492788;
                }
              }
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.997515678405763495) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.924581527709961826) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.497866153717041238) ) ) {
                    result[0] += 0.01171342712844357;
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.501469135284425604) ) ) {
                      if ( LIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)48.00000000000000711) ) ) {
                        result[0] += -0.04020073545295487;
                      } else {
                        result[0] += -0.07887671775265792;
                      }
                    } else {
                      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += -0.06969869544311881;
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.737386107444763628) ) ) {
                          result[0] += 0.0400092328934566;
                        } else {
                          result[0] += -0.036910058316532206;
                        }
                      }
                    }
                  }
                } else {
                  result[0] += 0.003296437193432883;
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.737386107444763628) ) ) {
                  result[0] += 0.0310179471242317;
                } else {
                  if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)23.50000000000000355) ) ) {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.216319084167481357) ) ) {
                      result[0] += -0.04313523467738996;
                    } else {
                      result[0] += 0.02065283244321691;
                    }
                  } else {
                    result[0] += 0.023049647030307745;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.744487762451173651) ) ) {
              if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.08026343505374786;
              } else {
                result[0] += -0.025547406611270564;
              }
            } else {
              result[0] += -0.011501851613211904;
            }
          }
        } else {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.90474271774292081) ) ) {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)12.50000000000000178) ) ) {
              result[0] += 0.013725811790320182;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.649621725082398349) ) ) {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
                  if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                    result[0] += 0.010023002439276346;
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.56941866874694913) ) ) {
                      result[0] += -0.02994541663192491;
                    } else {
                      result[0] += -0.09509597228844041;
                    }
                  }
                } else {
                  result[0] += 0.010293827445097183;
                }
              } else {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.537586927413940874) ) ) {
                  result[0] += -0.00017531841849587523;
                } else {
                  result[0] += 0.04644131301828122;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.321723937988282138) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.846404790878296787) ) ) {
                result[0] += -0.002229799344089813;
              } else {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                  result[0] += -0.03128914950247079;
                } else {
                  result[0] += -0.07300567809172166;
                }
              }
            } else {
              result[0] += 0.01383192592058753;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.342454433441162998) ) ) {
          if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
            result[0] += 0.0010245384183067633;
          } else {
            if ( UNLIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.01933870431018633;
            } else {
              result[0] += -0.06395351990838555;
            }
          }
        } else {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.2121162414550799) ) ) {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
              if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
                if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.772694945335388628) ) ) {
                        result[0] += 0.03281633594822483;
                      } else {
                        result[0] += 0.0034829068257106474;
                      }
                    } else {
                      if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)10.95635986328125178) ) ) {
                        result[0] += -0.014496295394267234;
                      } else {
                        result[0] += 0.06282880237708607;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
                      result[0] += -0.04809267767284616;
                    } else {
                      if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.534971714019776279) ) ) {
                          if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                            result[0] += -0.0020718194029793594;
                          } else {
                            result[0] += -0.05505215695417756;
                          }
                        } else {
                          result[0] += 0.020546335999007043;
                        }
                      } else {
                        result[0] += 0.03383730798779049;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)15.50000000000000178) ) ) {
                    if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)3.500000000000000444) ) ) {
                      result[0] += 0.023822288530105137;
                    } else {
                      result[0] += -0.022631865921100936;
                    }
                  } else {
                    if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)20.50000000000000355) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.53326439857482999) ) ) {
                        result[0] += 0.015184695174901345;
                      } else {
                        result[0] += 0.09110620176779098;
                      }
                    } else {
                      result[0] += 0.01930103596788766;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)11.09085798263549982) ) ) {
                  result[0] += 0.024196111139022456;
                } else {
                  result[0] += -0.16250406102643578;
                }
              }
            } else {
              result[0] += -0.037334791065564164;
            }
          } else {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
              if ( LIKELY( !(data[56].missing != -1) || (data[56].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += -0.021379060602076408;
              } else {
                result[0] += 0.0204326945638023;
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.53326439857482999) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.08656392363906659;
                  } else {
                    result[0] += -0.03594769791738731;
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.397998809814454013) ) ) {
                    result[0] += -0.0943452970706696;
                  } else {
                    result[0] += 0.014401877379103182;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)15.50000000000000178) ) ) {
                  result[0] += 0.12647330067556076;
                } else {
                  result[0] += 0.05213868357028243;
                }
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)28.50000000000000355) ) ) {
        if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
          result[0] += 0.015490427224486276;
        } else {
          result[0] += -0.005349421064074308;
        }
      } else {
        result[0] += 5.268043342898583e-05;
      }
    }
  }
  if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.909855604171753818) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.97887301445007413) ) ) {
          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
            result[0] += 0.07363937864734309;
          } else {
            if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += 0.015243385918154315;
            } else {
              result[0] += -0.06932226270789663;
            }
          }
        } else {
          if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.262283086776734287) ) ) {
              result[0] += -0.01109183543463275;
            } else {
              result[0] += -0.05947394805052352;
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.262283086776734287) ) ) {
              result[0] += -0.07198143218754836;
            } else {
              if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.05469019316463091;
              } else {
                result[0] += -0.043024354970328915;
              }
            }
          }
        }
      } else {
        result[0] += -0.055707321293873825;
      }
    } else {
      if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.868834793567657693) ) ) {
        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)1.242453336715698464) ) ) {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)6.500000000000000888) ) ) {
            if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)2.071567356586456743) ) ) {
              if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
                      if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += -0.010501505444917968;
                      } else {
                        result[0] += 0.053901055896561;
                      }
                    } else {
                      result[0] += -0.04442721018738286;
                    }
                  } else {
                    if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                      result[0] += 0.0002174834123495332;
                    } else {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.400584220886231357) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.815814018249513495) ) ) {
                          result[0] += 0.004358148146334299;
                        } else {
                          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.662244915962219682) ) ) {
                            result[0] += 0.00756947446946092;
                          } else {
                            result[0] += -0.040472524444625756;
                          }
                        }
                      } else {
                        if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)243.5000000000000284) ) ) {
                          result[0] += 0.015259495845887107;
                        } else {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
                            result[0] += 0.00683245705813439;
                          } else {
                            result[0] += -0.021165325563671757;
                          }
                        }
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.932935476303101474) ) ) {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.384830474853516513) ) ) {
                      if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                        result[0] += -0.015630109039915196;
                      } else {
                        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                          result[0] += 0.022379153289678193;
                        } else {
                          result[0] += -0.024066854938738794;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)5.500000000000000888) ) ) {
                        result[0] += 0.02376641217121361;
                      } else {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.69067406654357999) ) ) {
                          result[0] += -0.14056634002048582;
                        } else {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.921924352645874468) ) ) {
                            result[0] += 0.020387976279601955;
                          } else {
                            result[0] += -0.054626843150275065;
                          }
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.198252916336060458) ) ) {
                      result[0] += -0.0029738781116369845;
                    } else {
                      if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += -0.0667072998607793;
                      } else {
                        result[0] += -0.015906023165693343;
                      }
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.835998296737671787) ) ) {
                    result[0] += -0.0023641103700763025;
                  } else {
                    if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += -0.047612748782471515;
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.264029741287232333) ) ) {
                          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)167.5000000000000284) ) ) {
                            result[0] += -0.04860948155523059;
                          } else {
                            result[0] += -0.011953880272260702;
                          }
                        } else {
                          if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                            result[0] += -0.04239082989207875;
                          } else {
                            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.587308406829834873) ) ) {
                                result[0] += -0.06110674584991595;
                              } else {
                                result[0] += -0.004111982137592007;
                              }
                            } else {
                              result[0] += 0.013479678023175054;
                            }
                          }
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.497866153717041238) ) ) {
                        result[0] += 0.01414962830595824;
                      } else {
                        result[0] += -0.06241880589255956;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.581200122833253729) ) ) {
                    if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.828941345214844638) ) ) {
                          result[0] += 0.021355738518444613;
                        } else {
                          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.802696108818054643) ) ) {
                            result[0] += 0.07413420844046363;
                          } else {
                            result[0] += -0.02367744183494716;
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.868834793567657693) ) ) {
                          result[0] += 0.0029226498983832283;
                        } else {
                          result[0] += 0.02053867786837481;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
                        result[0] += 0.003159227916453862;
                      } else {
                        result[0] += -0.043552720953295446;
                      }
                    }
                  } else {
                    result[0] += -0.02163709103528963;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.0356345599583439;
              } else {
                result[0] += 0.037408036417167216;
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.901921629905701128) ) ) {
              result[0] += 0.021385435719621825;
            } else {
              result[0] += -0.04462917801583184;
            }
          }
        } else {
          result[0] += 0.08693878601204467;
        }
      } else {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)6.500000000000000888) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.59600305557251154) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.53326439857482999) ) ) {
              result[0] += -0.0001905653903878691;
            } else {
              result[0] += -0.04697617004229973;
            }
          } else {
            if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                result[0] += -0.05459365699522879;
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.531673669815064365) ) ) {
                  result[0] += -0.04511525819200155;
                } else {
                  result[0] += 0.09815005500813814;
                }
              }
            } else {
              if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)6.455969333648682529) ) ) {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
                  if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += 0.18405063373932531;
                  } else {
                    result[0] += -0.03820674405403909;
                  }
                } else {
                  result[0] += 0.08198370226686905;
                }
              } else {
                result[0] += -0.08115439676263478;
              }
            }
          }
        } else {
          result[0] += 0.11928754597076155;
        }
      }
    }
  } else {
    result[0] += 0.000280589169090963;
  }
  if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
    result[0] += -0.00027458316482913283;
  } else {
    if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.102609157562256748) ) ) {
      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)6.245125532150269443) ) ) {
        if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)161.5000000000000284) ) ) {
            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.012675821781158891) ) ) {
              result[0] += -0.04857300481898975;
            } else {
              if ( UNLIKELY( !(data[56].missing != -1) || (data[56].fvalue <= (double)6.000000000000000888) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.510617971420288974) ) ) {
                  result[0] += 0.08119899895380024;
                } else {
                  result[0] += -0.009622404517649707;
                }
              } else {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)100.5000000000000142) ) ) {
                  result[0] += 0.002425011378347321;
                } else {
                  result[0] += 0.04239072403522779;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.761470437049866167) ) ) {
              result[0] += -0.001851959879380952;
            } else {
              result[0] += -0.10650533332552606;
            }
          }
        } else {
          if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)96.50000000000001421) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.088880300521851474) ) ) {
              result[0] += -0.026495911246563576;
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.04690074920654475) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.835998296737671787) ) ) {
                  result[0] += 0.10368132456615989;
                } else {
                  result[0] += -0.006235285815476052;
                }
              } else {
                result[0] += 0.11048617179668854;
              }
            }
          } else {
            result[0] += -3.351598652193352e-05;
          }
        }
      } else {
        if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)142.5000000000000284) ) ) {
          result[0] += 0.029507108933605693;
        } else {
          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)238.5000000000000284) ) ) {
            result[0] += -0.008798051570750996;
          } else {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)270.5000000000000568) ) ) {
              if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.700598716735840066) ) ) {
                  result[0] += 0.06835041746856052;
                } else {
                  result[0] += -0.057418486067037755;
                }
              } else {
                if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.021766088577683765;
                } else {
                  result[0] += 0.12114434509156309;
                }
              }
            } else {
              result[0] += -0.0064991446578618155;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
          result[0] += -0.00963434638323976;
        } else {
          if ( LIKELY( !(data[60].missing != -1) || (data[60].fvalue <= (double)6.000000000000000888) ) ) {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.802901029586792436) ) ) {
                result[0] += 0.02867728049655944;
              } else {
                result[0] += 0.06335030264431406;
              }
            } else {
              result[0] += 0.008913119082652046;
            }
          } else {
            result[0] += 0.007088020721016878;
          }
        }
      } else {
        if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.700598716735840066) ) ) {
            if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
              if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.241523027420044833) ) ) {
                  result[0] += 0.12591912101669703;
                } else {
                  result[0] += -0.00030195081962018557;
                }
              } else {
                result[0] += 0.016402632547663998;
              }
            } else {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.017318914081771032;
              } else {
                if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)1.242453336715698464) ) ) {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.537586927413940874) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.313699722290040839) ) ) {
                      result[0] += 0.019550978960068372;
                    } else {
                      result[0] += -0.018738121822145442;
                    }
                  } else {
                    result[0] += 0.016380065172566856;
                  }
                } else {
                  result[0] += -0.07607220030733443;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)156.5000000000000284) ) ) {
                  if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)144.5000000000000284) ) ) {
                    if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)142.5000000000000284) ) ) {
                      if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.32411074638366788) ) ) {
                        result[0] += -0.047241005593882335;
                      } else {
                        result[0] += 0.025990031654593557;
                      }
                    } else {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.321723937988282138) ) ) {
                        result[0] += -0.09002233379557636;
                      } else {
                        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += 0.09422634644339219;
                        } else {
                          result[0] += -0.05290016398824824;
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.7592954635620135) ) ) {
                      result[0] += 0.084466689160232;
                    } else {
                      result[0] += 0.02020721884081472;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)9.488236904144288886) ) ) {
                      if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)298.5000000000000568) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.05835151672363459) ) ) {
                          result[0] += -0.11904766978640116;
                        } else {
                          result[0] += 0.04745439341071936;
                        }
                      } else {
                        result[0] += -0.007477830722983281;
                      }
                    } else {
                      result[0] += -0.027331666806492894;
                    }
                  } else {
                    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)7.232009172439576083) ) ) {
                          if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)276.5000000000000568) ) ) {
                            result[0] += 0.0716700615625556;
                          } else {
                            result[0] += 0.0185710174284088;
                          }
                        } else {
                          result[0] += 0.01318637090681785;
                        }
                      } else {
                        result[0] += -0.00629156999694826;
                      }
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.765202045440675604) ) ) {
                        result[0] += 0.0827798545426566;
                      } else {
                        result[0] += -0.008487250774416097;
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.0849568382987671;
                } else {
                  if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.321723937988282138) ) ) {
                        result[0] += -0.11252749238611835;
                      } else {
                        result[0] += -0.00024118141745221178;
                      }
                    } else {
                      result[0] += 0.04599954009022136;
                    }
                  } else {
                    if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.700598716735840066) ) ) {
                      result[0] += -0.10616963930350579;
                    } else {
                      result[0] += 0.03347919519313619;
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                result[0] += -0.06748904194753098;
              } else {
                result[0] += 0.0009419000989340332;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
            result[0] += -0.02083794562354505;
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.241523027420044833) ) ) {
              result[0] += -0.018024797517098273;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.61675357818603693) ) ) {
                if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                  result[0] += -0.044652614985718646;
                } else {
                  result[0] += 0.025124871809310556;
                }
              } else {
                result[0] += 0.06030280498975308;
              }
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
      result[0] += -0.017533658780100266;
    } else {
      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)14.98329114913940607) ) ) {
        if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.531007289886475498) ) ) {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.863673448562622958) ) ) {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.153024196624756748) ) ) {
                    result[0] += 0.004622911962963933;
                  } else {
                    result[0] += -0.020823255227785584;
                  }
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.219419956207276279) ) ) {
                    result[0] += 0.018827300728593763;
                  } else {
                    result[0] += 0.004682581818377413;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)36.50000000000000711) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.040618419647218573) ) ) {
                    if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
                      result[0] += 0.1061073873553118;
                    } else {
                      result[0] += 0.007236965920615001;
                    }
                  } else {
                    result[0] += 0.0024705128801126575;
                  }
                } else {
                  result[0] += -0.017216992040978898;
                }
              }
            } else {
              result[0] += -0.02578405550040476;
            }
          } else {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.216319084167481357) ) ) {
              result[0] += -0.00416596954807844;
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)5.500000000000000888) ) ) {
                result[0] += 0.004588880069603477;
              } else {
                result[0] += -0.05909520723735543;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.835998296737671787) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.126503944396974433) ) ) {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)45.50000000000000711) ) ) {
                    result[0] += -0.04490050086571066;
                  } else {
                    result[0] += -0.012511854792552622;
                  }
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.53326439857482999) ) ) {
                    result[0] += -0.02037398511001453;
                  } else {
                    if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)48.00000000000000711) ) ) {
                        result[0] += 0.032784874275633806;
                      } else {
                        result[0] += -0.005442379526550287;
                      }
                    } else {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.474771499633789951) ) ) {
                        result[0] += -0.018815886661671333;
                      } else {
                        result[0] += 0.03270960167706815;
                      }
                    }
                  }
                }
              } else {
                result[0] += 0.000987765346111339;
              }
            } else {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
                  result[0] += -0.05032533590957186;
                } else {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                      result[0] += -0.07287640483554682;
                    } else {
                      result[0] += -0.024536283696475574;
                    }
                  } else {
                    if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
                      result[0] += -0.020615081765515568;
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.596743106842042792) ) ) {
                        result[0] += -0.03809351059858324;
                      } else {
                        result[0] += 0.03690875987998191;
                      }
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.497866153717041238) ) ) {
                  if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)252.5000000000000284) ) ) {
                    if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)47.50000000000000711) ) ) {
                      result[0] += 0.027794316749376654;
                    } else {
                      result[0] += -0.02006259730854283;
                    }
                  } else {
                    result[0] += 0.0524817599443913;
                  }
                } else {
                  result[0] += -0.06344667518832285;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.744487762451173651) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.778982400894165927) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.338562726974488193) ) ) {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
                    result[0] += 0.007994806431769592;
                  } else {
                    result[0] += 0.08082530859867812;
                  }
                } else {
                  if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += 0.036696492304561096;
                  } else {
                    result[0] += 0.08119300001598329;
                  }
                }
              } else {
                if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.288254261016846591) ) ) {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)5.500000000000000888) ) ) {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.56941866874694913) ) ) {
                        result[0] += 0.015334314970467;
                      } else {
                        result[0] += -0.03876627190825262;
                      }
                    } else {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.605120182037354404) ) ) {
                        result[0] += 0.05878757494823849;
                      } else {
                        result[0] += 0.01667949784821783;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)46.50000000000000711) ) ) {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.56941866874694913) ) ) {
                        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.53326439857482999) ) ) {
                          result[0] += 0.022058291074057483;
                        } else {
                          result[0] += -0.019943884785910998;
                        }
                      } else {
                        result[0] += 0.018108257051744927;
                      }
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.649621725082398349) ) ) {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.737386107444763628) ) ) {
                          result[0] += 0.05188099737103587;
                        } else {
                          result[0] += -0.003297805375160209;
                        }
                      } else {
                        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.267844915390015537) ) ) {
                          result[0] += -0.007227642956136921;
                        } else {
                          result[0] += -0.03098862512373736;
                        }
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.590985536575318271) ) ) {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.56941866874694913) ) ) {
                        result[0] += -0.021045404659672834;
                      } else {
                        result[0] += -0.10304707517217403;
                      }
                    } else {
                      result[0] += 0.017772981610663027;
                    }
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.450390577316285068) ) ) {
                      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
                        result[0] += 0.028584337898313696;
                      } else {
                        result[0] += -0.035051355251233825;
                      }
                    } else {
                      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)5.500000000000000888) ) ) {
                        result[0] += 0.07249334144275095;
                      } else {
                        result[0] += -0.02584630441561335;
                      }
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
                result[0] += -0.07777527463056959;
              } else {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)5.500000000000000888) ) ) {
                  if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.342454433441162998) ) ) {
                        result[0] += 0.026093051792209006;
                      } else {
                        result[0] += -0.07748850098233173;
                      }
                    } else {
                      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.174569487571716753) ) ) {
                        if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)167.5000000000000284) ) ) {
                          result[0] += 0.008192941676737514;
                        } else {
                          result[0] += -0.039732104866113165;
                        }
                      } else {
                        result[0] += 0.0674190862351862;
                      }
                    }
                  } else {
                    result[0] += -0.04763006409580828;
                  }
                } else {
                  result[0] += -0.06365033409849617;
                }
              }
            }
          }
        }
      } else {
        result[0] += -0.015349504777838991;
      }
    }
  } else {
    result[0] += 0.00025763091727653865;
  }
  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.37109279632568537) ) ) {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.342454433441162998) ) ) {
        result[0] += 0.017159969348263866;
      } else {
        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.264029741287232333) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.241523027420044833) ) ) {
              result[0] += -0.003062175954134979;
            } else {
              if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.95797300338745206) ) ) {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.255632162094117099) ) ) {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
                      if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)91.50000000000001421) ) ) {
                        if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.156774044036865678) ) ) {
                          result[0] += -0.060609146415865234;
                        } else {
                          result[0] += -0.003604347057551813;
                        }
                      } else {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                          result[0] += 0.01152041201437651;
                        } else {
                          result[0] += -0.03612650022400317;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.138333082199097124) ) ) {
                        result[0] += -0.014869189675355663;
                      } else {
                        result[0] += 0.025664342333887064;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += 0.029494594694965154;
                    } else {
                      result[0] += 0.0071108084893905625;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.97887301445007413) ) ) {
                      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += 0.11751763876064203;
                      } else {
                        result[0] += 0.0486362201421271;
                      }
                    } else {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.216319084167481357) ) ) {
                        result[0] += 0.0623768525755477;
                      } else {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.921100616455079013) ) ) {
                          result[0] += -0.07177381795008653;
                        } else {
                          result[0] += 0.0534677174074792;
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.102759599685669833) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.921924352645874468) ) ) {
                          result[0] += 0.03486863698134062;
                        } else {
                          if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                            if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                              result[0] += -0.04580295694762979;
                            } else {
                              result[0] += -0.11800809766828434;
                            }
                          } else {
                            result[0] += 0.027044385106593096;
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.329314231872559482) ) ) {
                          result[0] += 0.07344788486504056;
                        } else {
                          if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)92.50000000000001421) ) ) {
                            result[0] += -0.08409084122427399;
                          } else {
                            result[0] += 0.0318596575678246;
                          }
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)2.970085620880127397) ) ) {
                        result[0] += 0.08582807429716349;
                      } else {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.216319084167481357) ) ) {
                          if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.610145330429078037) ) ) {
                            result[0] += 0.04558448824843836;
                          } else {
                            result[0] += -0.05623140143764802;
                          }
                        } else {
                          result[0] += -0.013503173975210945;
                        }
                      }
                    }
                  }
                }
              } else {
                result[0] += -0.13219375808035388;
              }
            }
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.539549827575684482) ) ) {
                if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.158952236175537998) ) ) {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.23832273483276456) ) ) {
                      if ( UNLIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)6.000000000000000888) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.744487762451173651) ) ) {
                          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.029068946838379794) ) ) {
                            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
                              result[0] += -0.11760417618221153;
                            } else {
                              result[0] += -0.03738225753081969;
                            }
                          } else {
                            if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.650573849678039995) ) ) {
                              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.917405366897583452) ) ) {
                                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.249904870986938921) ) ) {
                                  result[0] += 0.004426509828199332;
                                } else {
                                  result[0] += 0.076188120837943;
                                }
                              } else {
                                result[0] += -0.03296869042087613;
                              }
                            } else {
                              result[0] += -0.06305289417674448;
                            }
                          }
                        } else {
                          result[0] += 0.015358464495330204;
                        }
                      } else {
                        result[0] += 0.011858106242382052;
                      }
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.285887241363526279) ) ) {
                        result[0] += 0.0445586552402057;
                      } else {
                        result[0] += 0.015487589403811359;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.780892848968506748) ) ) {
                      result[0] += 0.0002975818620706619;
                    } else {
                      result[0] += -0.023034188212770732;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)231.5000000000000284) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.141444921493531162) ) ) {
                      result[0] += -0.018675801394839996;
                    } else {
                      result[0] += -0.051963725205561584;
                    }
                  } else {
                    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                      result[0] += 0.01747512169374189;
                    } else {
                      result[0] += -0.02847304998290716;
                    }
                  }
                }
              } else {
                result[0] += -0.03048780332092987;
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.216319084167481357) ) ) {
                if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)7.933616161346436435) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.338562726974488193) ) ) {
                    result[0] += 0.09322345520824595;
                  } else {
                    result[0] += -0.0008430596465543419;
                  }
                } else {
                  result[0] += -0.0470102613271714;
                }
              } else {
                result[0] += -0.056679017391243994;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.921100616455079013) ) ) {
            if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)7.488013744354248935) ) ) {
              result[0] += 0.007162276672427541;
            } else {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.216319084167481357) ) ) {
                  result[0] += 0.02513034186767593;
                } else {
                  result[0] += 0.07150591210590833;
                }
              } else {
                result[0] += -0.0016691948075058388;
              }
            }
          } else {
            if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)7.817222595214844638) ) ) {
              result[0] += 0.005189905593901228;
            } else {
              result[0] += -0.04037350865388278;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.662244915962219682) ) ) {
        result[0] += -0.049222053470248134;
      } else {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)321.5000000000000568) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
              if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += 0.023765865590570243;
              } else {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.0347667984892141;
                } else {
                  result[0] += 0.016175228897721102;
                }
              }
            } else {
              result[0] += -0.04261037448543957;
            }
          } else {
            result[0] += 0.035374537524766866;
          }
        } else {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.344550132751465732) ) ) {
            result[0] += -0.010750215767863827;
          } else {
            result[0] += -0.04360473831132672;
          }
        }
      }
    }
  } else {
    result[0] += -0.0001759336441280164;
  }
  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.24049568176269709) ) ) {
      if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.318498134613038886) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.285887241363526279) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.679712533950806552) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.289602279663086826) ) ) {
                    result[0] += -0.0070442416433105825;
                  } else {
                    result[0] += 0.048919927619486936;
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.142630577087403232) ) ) {
                    result[0] += 0.03529021174444535;
                  } else {
                    if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.835998296737671787) ) ) {
                        result[0] += -0.12124896316552705;
                      } else {
                        result[0] += -0.04075386088491148;
                      }
                    } else {
                      result[0] += -0.0008700266124975763;
                    }
                  }
                }
              } else {
                result[0] += 0.018598333674305623;
              }
            } else {
              result[0] += 0.015000859649890062;
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.623839378356934482) ) ) {
              result[0] += 0.022904375629368624;
            } else {
              result[0] += 0.083465285010548;
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.264029741287232333) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.241523027420044833) ) ) {
              result[0] += -0.002726478257344957;
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)2.970085620880127397) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.780892848968506748) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                      result[0] += -0.03237396107495236;
                    } else {
                      result[0] += 0.06087705267103025;
                    }
                  } else {
                    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                      result[0] += 0.029919209930220587;
                    } else {
                      result[0] += -0.11225134640575514;
                    }
                  }
                } else {
                  result[0] += 0.010540669531328407;
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)2.970085620880127397) ) ) {
                  result[0] += 0.10602621402731938;
                } else {
                  if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)91.50000000000001421) ) ) {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.924581527709961826) ) ) {
                      result[0] += 0.06434313832144792;
                    } else {
                      result[0] += -0.035497918564028993;
                    }
                  } else {
                    if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)92.50000000000001421) ) ) {
                      result[0] += -0.07426513424788038;
                    } else {
                      result[0] += 0.0071701239056347725;
                    }
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.53326439857482999) ) ) {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.008295540663692117;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.289602279663086826) ) ) {
                  result[0] += -0.14738302905387807;
                } else {
                  result[0] += -0.007579209705955978;
                }
              }
            } else {
              if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.873467922210695136) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.216319084167481357) ) ) {
                    result[0] += 0.008370790824705729;
                  } else {
                    result[0] += -0.016416236661551713;
                  }
                } else {
                  if ( UNLIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)6.000000000000000888) ) ) {
                    if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                      result[0] += -0.08295938377313217;
                    } else {
                      result[0] += 0.007065293212588398;
                    }
                  } else {
                    result[0] += -0.038158184751866775;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)92.50000000000001421) ) ) {
                  result[0] += -0.0698025492362294;
                } else {
                  if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += -0.05822005600619765;
                  } else {
                    result[0] += 0.016462972459857932;
                  }
                }
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)225.5000000000000284) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.601370334625245029) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.53326439857482999) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.43749904632568537) ) ) {
                result[0] += 0.0024783508295370214;
              } else {
                result[0] += 0.07042226410283352;
              }
            } else {
              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                    result[0] += 0.022460863970837505;
                  } else {
                    result[0] += -0.04331284989496956;
                  }
                } else {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.835998296737671787) ) ) {
                      result[0] += -0.08849579446168593;
                    } else {
                      result[0] += 0.04946766706694885;
                    }
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.216319084167481357) ) ) {
                      result[0] += 0.07262296138693856;
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.190353393554689276) ) ) {
                        result[0] += 0.015681207693708316;
                      } else {
                        result[0] += -0.08014447549567759;
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.649621725082398349) ) ) {
                  if ( UNLIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.06748051343269572;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.288254261016846591) ) ) {
                      result[0] += -0.015625935667288062;
                    } else {
                      if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)4.749261140823365146) ) ) {
                        result[0] += 0.11500564510349794;
                      } else {
                        result[0] += 0.03767384876422212;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)87.50000000000001421) ) ) {
                    result[0] += 0.05033653754712796;
                  } else {
                    result[0] += -0.005450038021308281;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.539540290832521308) ) ) {
              result[0] += 0.00905804649163239;
            } else {
              result[0] += -0.051665266040095074;
            }
          }
        } else {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.09369468688965021) ) ) {
            result[0] += -0.0006320207108569116;
          } else {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.09271012906459931;
            } else {
              result[0] += -0.015204851841436332;
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.780892848968506748) ) ) {
          if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.662244915962219682) ) ) {
              result[0] += -0.059695015992086056;
            } else {
              result[0] += 0.019281192807603747;
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.835998296737671787) ) ) {
              result[0] += -0.05832644236925203;
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.241523027420044833) ) ) {
                result[0] += 0.026071239583694762;
              } else {
                result[0] += -0.016901942358877126;
              }
            }
          }
        } else {
          result[0] += -0.052283738438587;
        }
      } else {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.637949228286744052) ) ) {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.802696108818054643) ) ) {
            result[0] += -0.04327604845233337;
          } else {
            if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
              result[0] += -0.0054268124672639615;
            } else {
              result[0] += -0.04256494026671584;
            }
          }
        } else {
          result[0] += -0.054553865259364835;
        }
      }
    }
  } else {
    result[0] += -0.00019935711001849298;
  }
  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.24049568176269709) ) ) {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)2.012675821781158891) ) ) {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)231.5000000000000284) ) ) {
            result[0] += -0.016082401896956026;
          } else {
            result[0] += 0.04155343798555987;
          }
        } else {
          result[0] += 0.09773794314025358;
        }
      } else {
        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( LIKELY( !(data[10].missing != -1) || (data[10].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
              if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)6.000000000000000888) ) ) {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.008262563294587936;
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
                      result[0] += 0.00298742077097246;
                    } else {
                      if ( UNLIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += 0.010413067334767863;
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.581200122833253729) ) ) {
                          result[0] += 0.06970089929975855;
                        } else {
                          result[0] += 0.019890360391129666;
                        }
                      }
                    }
                  } else {
                    result[0] += 0.04356341221052487;
                  }
                }
              } else {
                if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.472188472747804511) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.285887241363526279) ) ) {
                    if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.015471313676349827;
                    } else {
                      result[0] += 0.007852074252536837;
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.590985536575318271) ) ) {
                      result[0] += 0.09880602519554343;
                    } else {
                      result[0] += 0.013578134415287976;
                    }
                  }
                } else {
                  result[0] += 0.0462510010941314;
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.264029741287232333) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.241523027420044833) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.986973047256470615) ) ) {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.518026351928711826) ) ) {
                      result[0] += -0.01911383681856023;
                    } else {
                      result[0] += 0.0682060622403406;
                    }
                  } else {
                    result[0] += 0.0010826316168697646;
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.241523027420044833) ) ) {
                    result[0] += -0.046330760472511;
                  } else {
                    if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.497866153717041238) ) ) {
                      result[0] += 0.011516608535315465;
                    } else {
                      result[0] += -0.025965518056965915;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.539549827575684482) ) ) {
                    if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                      result[0] += 0.001867878187463954;
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.141444921493531162) ) ) {
                        if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)6.000000000000000888) ) ) {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.190353393554689276) ) ) {
                            if ( UNLIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                              result[0] += -0.041259089843842064;
                            } else {
                              result[0] += 0.026662657640270034;
                            }
                          } else {
                            result[0] += -0.034353223127001176;
                          }
                        } else {
                          result[0] += 0.03036302561391755;
                        }
                      } else {
                        result[0] += -0.0482849224248576;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)89.50000000000001421) ) ) {
                      result[0] += -0.003036196196771905;
                    } else {
                      result[0] += -0.04284489853626208;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.623839378356934482) ) ) {
                    if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += -0.05258898348804373;
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.909102678298951083) ) ) {
                        result[0] += 0.014496386128931355;
                      } else {
                        result[0] += -0.029825559880284916;
                      }
                    }
                  } else {
                    result[0] += -0.07167230050075246;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.649621725082398349) ) ) {
              result[0] += 0.06718306844790951;
            } else {
              result[0] += -0.03740782010637897;
            }
          }
        } else {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
            if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.921100616455079013) ) ) {
                if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.701225757598877397) ) ) {
                  result[0] += 0.015334624728176635;
                } else {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.190353393554689276) ) ) {
                      result[0] += 0.00897888457489812;
                    } else {
                      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.040618419647218573) ) ) {
                          result[0] += 0.012618092714487992;
                        } else {
                          result[0] += 0.07545489880582758;
                        }
                      } else {
                        result[0] += 0.09953120499183599;
                      }
                    }
                  } else {
                    result[0] += 0.02569373848313121;
                  }
                }
              } else {
                result[0] += -0.014922714890093567;
              }
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.474771499633789951) ) ) {
                  result[0] += -0.03945044349368927;
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.835998296737671787) ) ) {
                    result[0] += 0.09925458678992216;
                  } else {
                    if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.166635274887085849) ) ) {
                      result[0] += 0.0067215482373696265;
                    } else {
                      result[0] += -0.06192584553368561;
                    }
                  }
                }
              } else {
                result[0] += 0.02599963642773301;
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.318498134613038886) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.342765808105469638) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.216319084167481357) ) ) {
                  if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.288254261016846591) ) ) {
                      result[0] += -0.07513634567638805;
                    } else {
                      result[0] += 5.237551603563674e-05;
                    }
                  } else {
                    result[0] += 0.0595680703167765;
                  }
                } else {
                  result[0] += 0.03964397013457647;
                }
              } else {
                result[0] += -0.07418306533119033;
              }
            } else {
              result[0] += -0.05121403380541328;
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.780892848968506748) ) ) {
          if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.662244915962219682) ) ) {
              result[0] += -0.04963835331004963;
            } else {
              result[0] += 0.016429575707161786;
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.835998296737671787) ) ) {
              result[0] += -0.055256650896021345;
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.53326439857482999) ) ) {
                result[0] += 0.023859818997003273;
              } else {
                result[0] += -0.015881227056287205;
              }
            }
          }
        } else {
          result[0] += -0.047298798579900284;
        }
      } else {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.637949228286744052) ) ) {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.802696108818054643) ) ) {
            result[0] += -0.03927456326155798;
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.241523027420044833) ) ) {
              result[0] += 0.000902808884994099;
            } else {
              result[0] += -0.02387002950587171;
            }
          }
        } else {
          result[0] += -0.049326792662197534;
        }
      }
    }
  } else {
    result[0] += -0.00020300415025511327;
  }
  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.085941076278687412) ) ) {
      if ( LIKELY( !(data[10].missing != -1) || (data[10].fvalue <= (double)0.8958797454833985485) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)2.012675821781158891) ) ) {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
            result[0] += 0.008342638217016512;
          } else {
            result[0] += 0.09577920446521186;
          }
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.714269638061524326) ) ) {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.342454433441162998) ) ) {
                result[0] += 0.01185696146153877;
              } else {
                if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.684611082077027255) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.029068946838379794) ) ) {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.673553824424744096) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.397998809814454013) ) ) {
                        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                          result[0] += -0.056580898760708334;
                        } else {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.649621725082398349) ) ) {
                            result[0] += -0.07016440102557524;
                          } else {
                            result[0] += 0.037688552675205975;
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                          result[0] += 0.018888749986015377;
                        } else {
                          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.590985536575318271) ) ) {
                            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.623839378356934482) ) ) {
                              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
                                result[0] += -0.02019264048203395;
                              } else {
                                result[0] += -0.10353116662396343;
                              }
                            } else {
                              result[0] += -0.00123619449369818;
                            }
                          } else {
                            result[0] += 0.007738378396656274;
                          }
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += 0.01775464962407976;
                      } else {
                        result[0] += -0.007804167540718953;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.068990230560303623) ) ) {
                      result[0] += 0.005648170858293333;
                    } else {
                      result[0] += 0.09650225339236268;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.693369150161744052) ) ) {
                      result[0] += 0.0112860424689623;
                    } else {
                      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += -0.00834219448451355;
                      } else {
                        result[0] += -0.042770164171125336;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.982575893402101386) ) ) {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.439558982849121982) ) ) {
                          result[0] += 0.054334404503112624;
                        } else {
                          result[0] += -0.01492446195740995;
                        }
                      } else {
                        result[0] += -0.05577761388982644;
                      }
                    } else {
                      result[0] += -0.04966807912132671;
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.623839378356934482) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.795762062072754794) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.43749904632568537) ) ) {
                    if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += 0.03843292911774163;
                    } else {
                      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                        if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                          if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.03420138359069913) ) ) {
                            result[0] += -0.08397605134217313;
                          } else {
                            result[0] += 0.02564605653242505;
                          }
                        } else {
                          result[0] += -0.01228048094990855;
                        }
                      } else {
                        if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.417592287063599077) ) ) {
                          if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.285735368728638583) ) ) {
                            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.556798219680787021) ) ) {
                              result[0] += -0.0577861835493566;
                            } else {
                              if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.233775377273560458) ) ) {
                                result[0] += 0.030201048018769702;
                              } else {
                                result[0] += -0.03187275193932116;
                              }
                            }
                          } else {
                            result[0] += 0.03061488613401993;
                          }
                        } else {
                          result[0] += -0.03706159107376133;
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.88024568557739435) ) ) {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.553655147552491123) ) ) {
                        result[0] += 0.08410446624856908;
                      } else {
                        if ( UNLIKELY( !(data[60].missing != -1) || (data[60].fvalue <= (double)6.000000000000000888) ) ) {
                          if ( UNLIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.637949228286744052) ) ) {
                              if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                                result[0] += 0.057664215258816466;
                              } else {
                                result[0] += -0.05230047373528629;
                              }
                            } else {
                              result[0] += 0.0636789149209781;
                            }
                          } else {
                            result[0] += 0.027954825924664802;
                          }
                        } else {
                          if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                              result[0] += 0.043792666382163614;
                            } else {
                              result[0] += 0.1208183543708079;
                            }
                          } else {
                            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                              result[0] += 0.03558969406312978;
                            } else {
                              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.835998296737671787) ) ) {
                                result[0] += 0.09776548266320736;
                              } else {
                                result[0] += -0.034243614313102924;
                              }
                            }
                          }
                        }
                      }
                    } else {
                      result[0] += -0.0019099845407385537;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.198464870452881303) ) ) {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.33322238922119318) ) ) {
                        result[0] += 0.038220029798160554;
                      } else {
                        result[0] += -0.06123630959364227;
                      }
                    } else {
                      result[0] += -0.033613920035028944;
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.674522399902344638) ) ) {
                      if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.740319490432739702) ) ) {
                        result[0] += 0.0553519061650725;
                      } else {
                        result[0] += -0.03481134451881664;
                      }
                    } else {
                      result[0] += -0.06170047873813275;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)7.817222595214844638) ) ) {
                  if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)326.5000000000000568) ) ) {
                    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.0014339309550417078;
                    } else {
                      result[0] += 0.040955855198689384;
                    }
                  } else {
                    result[0] += -0.035503056283790684;
                  }
                } else {
                  if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += -0.014905299990859289;
                  } else {
                    result[0] += -0.08319856891442463;
                  }
                }
              }
            }
          } else {
            result[0] += -0.07278054159473192;
          }
        }
      } else {
        result[0] += -0.028175699665367166;
      }
    } else {
      if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.102609157562256748) ) ) {
          if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.350240230560303178) ) ) {
            result[0] += -0.12962876093298217;
          } else {
            result[0] += 0.031868965547881746;
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.255632162094117099) ) ) {
            result[0] += 0.007846865860832495;
          } else {
            result[0] += -0.007575241372604806;
          }
        }
      } else {
        if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.141444921493531162) ) ) {
            result[0] += -0.007727419418065664;
          } else {
            result[0] += -0.033982859774045415;
          }
        } else {
          result[0] += -0.048170482395226366;
        }
      }
    }
  } else {
    result[0] += -0.00019073749473571327;
  }
  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.24049568176269709) ) ) {
      if ( LIKELY( !(data[10].missing != -1) || (data[10].fvalue <= (double)0.8958797454833985485) ) ) {
        if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
            if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)6.000000000000000888) ) ) {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.280659198760987216) ) ) {
                  result[0] += -0.01340235308028329;
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
                    result[0] += 0.010000133982470183;
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.835998296737671787) ) ) {
                      result[0] += -0.05418077272108235;
                    } else {
                      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += 0.05323586903949569;
                      } else {
                        result[0] += -0.07533932350005557;
                      }
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
                  result[0] += 0.013485285365442943;
                } else {
                  result[0] += 0.045760983914850346;
                }
              }
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.623839378356934482) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.835998296737671787) ) ) {
                  if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.673553824424744096) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.025192260742188388) ) ) {
                      result[0] += -0.056235023803932915;
                    } else {
                      result[0] += 0.02106956150422293;
                    }
                  } else {
                    if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.627361297607423651) ) ) {
                      result[0] += 0.014872151276680899;
                    } else {
                      result[0] += 0.07170271481594669;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.53326439857482999) ) ) {
                      result[0] += -0.06769383592555273;
                    } else {
                      result[0] += -0.01927030007030873;
                    }
                  } else {
                    result[0] += 0.007487982497479633;
                  }
                }
              } else {
                result[0] += 0.038362757855812446;
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.264029741287232333) ) ) {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.605120182037354404) ) ) {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.667095184326172763) ) ) {
                      result[0] += -0.012949872965934278;
                    } else {
                      result[0] += 0.05635594155056071;
                    }
                  } else {
                    if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                      result[0] += 0.1272170986625746;
                    } else {
                      if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)3.276966691017151323) ) ) {
                        result[0] += -0.05750559893869213;
                      } else {
                        result[0] += 0.03390770191788909;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.644374847412110263) ) ) {
                    if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += 0.03901084430932925;
                    } else {
                      result[0] += 0.004821167383329806;
                    }
                  } else {
                    result[0] += 0.06176844626496584;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)2.970085620880127397) ) ) {
                  result[0] += 0.084350683281364;
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.216319084167481357) ) ) {
                    result[0] += 0.052298517294657114;
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.617236852645874912) ) ) {
                      result[0] += -0.025411351249018468;
                    } else {
                      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.329314231872559482) ) ) {
                          result[0] += 0.11352898439844449;
                        } else {
                          result[0] += 0.014793220721922929;
                        }
                      } else {
                        result[0] += -0.03669959769420152;
                      }
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)257681260544.0000305) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.029068946838379794) ) ) {
                  if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.497866153717041238) ) ) {
                    result[0] += -0.11956143047186361;
                  } else {
                    result[0] += 0.012918545153665629;
                  }
                } else {
                  if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
                    if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += -0.012392131411008582;
                    } else {
                      result[0] += -0.05158060362286268;
                    }
                  } else {
                    result[0] += -0.07015571874572163;
                  }
                }
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.851041555404663974) ) ) {
                  if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.397998809814454013) ) ) {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.249904870986938921) ) ) {
                        result[0] += -0.036527898719288236;
                      } else {
                        result[0] += 0.018592023677913933;
                      }
                    } else {
                      result[0] += 0.008606917688462026;
                    }
                  } else {
                    if ( UNLIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += 0.020249296915405113;
                    } else {
                      result[0] += -0.03093658026096945;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.921060562133789951) ) ) {
                      result[0] += -0.0015328810603354602;
                    } else {
                      result[0] += -0.03329506755925284;
                    }
                  } else {
                    result[0] += -0.04767891023121344;
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)9.899753093719484198) ) ) {
            if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)48.00000000000000711) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.617236852645874912) ) ) {
                result[0] += 0.019088888248856725;
              } else {
                result[0] += -0.017089137259288224;
              }
            } else {
              result[0] += 0.0024732986094216764;
            }
          } else {
            result[0] += 0.11404446108134365;
          }
        }
      } else {
        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.861792564392090288) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.649621725082398349) ) ) {
            result[0] += 0.06166307746162447;
          } else {
            result[0] += -0.05327759756961409;
          }
        } else {
          if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.540854334831238237) ) ) {
            result[0] += 0.04695621587779646;
          } else {
            result[0] += -0.08878433841853926;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.88435244560241788) ) ) {
          if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += 0.013019119354671056;
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.835998296737671787) ) ) {
              result[0] += -0.05236584893177011;
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.241523027420044833) ) ) {
                result[0] += 0.022865053355382026;
              } else {
                result[0] += -0.014823086727669445;
              }
            }
          }
        } else {
          result[0] += -0.04436296748716858;
        }
      } else {
        if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( UNLIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)6.000000000000000888) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.342454433441162998) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.802696108818054643) ) ) {
                result[0] += -0.03918836534323956;
              } else {
                result[0] += 0.04737240582962526;
              }
            } else {
              result[0] += -0.03212360405305711;
            }
          } else {
            result[0] += -0.0411324339803133;
          }
        } else {
          if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
              result[0] += 0.013362871819097783;
            } else {
              result[0] += -0.02817235304411785;
            }
          } else {
            result[0] += -0.024235764347376527;
          }
        }
      }
    }
  } else {
    result[0] += -0.0001909390256149854;
  }
  if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)6.000000000000000888) ) ) {
    result[0] += 0.00019499487269486354;
  } else {
    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
      if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)0.8958797454833985485) ) ) {
        if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)121.5000000000000142) ) ) {
          result[0] += -0.058421212337888544;
        } else {
          if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.556798219680787021) ) ) {
              result[0] += -0.040579143879345844;
            } else {
              result[0] += -0.0038542783127536425;
            }
          } else {
            if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)176.5000000000000284) ) ) {
                result[0] += 0.038266198899361004;
              } else {
                result[0] += -0.009088589697775235;
              }
            } else {
              result[0] += -0.024340482470060182;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)6144.000000000000909) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.028861761093140537) ) ) {
              result[0] += 0.009128300544770027;
            } else {
              result[0] += -0.02977680577266159;
            }
          } else {
            if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)273.5000000000000568) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.247576236724854404) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.071567356586456743) ) ) {
                  if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)272.5000000000000568) ) ) {
                    if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.998158693313599077) ) ) {
                      if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.182021141052246982) ) ) {
                        result[0] += -0.013140818156600507;
                      } else {
                        result[0] += -0.12278414250506725;
                      }
                    } else {
                      result[0] += 0.028665695910597033;
                    }
                  } else {
                    result[0] += -0.14948089009155835;
                  }
                } else {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.028266805314002087;
                    } else {
                      result[0] += 0.029701826265677148;
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.13022470474243342) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.131699204444885698) ) ) {
                        result[0] += 0.17370074614646505;
                      } else {
                        result[0] += 0.013837706979174073;
                      }
                    } else {
                      result[0] += -0.0297733734158753;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
                  result[0] += -0.024861745854084575;
                } else {
                  if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.877672910690308505) ) ) {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.20387363433838068) ) ) {
                        result[0] += -0.02641505566195268;
                      } else {
                        result[0] += 0.014395244730026239;
                      }
                    } else {
                      if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += -0.0746318868897958;
                      } else {
                        result[0] += 0.013167069750485441;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.249904870986938921) ) ) {
                      result[0] += 0.05539407613543138;
                    } else {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.131699204444885698) ) ) {
                        result[0] += 0.2115343006779844;
                      } else {
                        result[0] += 0.005275326756218061;
                      }
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
                result[0] += -0.0316321000912031;
              } else {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)280.5000000000000568) ) ) {
                  result[0] += -0.035347366973832374;
                } else {
                  result[0] += 0.0025171071661838403;
                }
              }
            }
          }
        } else {
          result[0] += -0.03097805624566759;
        }
      }
    } else {
      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.23832273483276456) ) ) {
        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.357691764831543413) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.556798219680787021) ) ) {
                result[0] += 0.042022282030733125;
              } else {
                result[0] += -0.005288570653969821;
              }
            } else {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.537586927413940874) ) ) {
                  if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                      if ( UNLIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += -0.08319640522477477;
                      } else {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.673553824424744096) ) ) {
                          result[0] += 0.008603572082391141;
                        } else {
                          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
                            result[0] += -0.026020893757994375;
                          } else {
                            result[0] += -0.07900192396900807;
                          }
                        }
                      }
                    } else {
                      result[0] += -0.02112256705063531;
                    }
                  } else {
                    if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)136.5000000000000284) ) ) {
                      result[0] += -0.032676595281544416;
                    } else {
                      result[0] += 0.0509421084866096;
                    }
                  }
                } else {
                  result[0] += -0.06470385254813905;
                }
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.30193185806274592) ) ) {
                  result[0] += -0.030612451454293862;
                } else {
                  result[0] += 0.02768863317970315;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.158010244369507724) ) ) {
                result[0] += 0.023722398875937117;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.594915628433228427) ) ) {
                  result[0] += 0.03383353655078637;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.513699531555176669) ) ) {
                    result[0] += -0.03716941188502126;
                  } else {
                    result[0] += -0.10335467971027529;
                  }
                }
              }
            } else {
              result[0] += 0.0014828604235547509;
            }
          }
        } else {
          if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)1.242453336715698464) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.761470437049866167) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)15.53430414199829279) ) ) {
                  if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)163.5000000000000284) ) ) {
                    result[0] += -0.004743673392737162;
                  } else {
                    if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)11.35262060165405451) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)2.970085620880127397) ) ) {
                        result[0] += -0.18970523024358665;
                      } else {
                        result[0] += 0.014700690642742504;
                      }
                    } else {
                      if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += 0.13321403446634572;
                      } else {
                        result[0] += -0.0006300825974038869;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.422742605209351474) ) ) {
                    result[0] += 0.08447500609779467;
                  } else {
                    result[0] += 0.014699024608015518;
                  }
                }
              } else {
                result[0] += -0.019523949296968435;
              }
            } else {
              result[0] += 0.08047608006275496;
            }
          } else {
            if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)2.012675821781158891) ) ) {
              if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.658699750900269443) ) ) {
                result[0] += -0.006337373106438404;
              } else {
                result[0] += -0.07536074399729227;
              }
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
                result[0] += -0.06498762508592386;
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)17.75587940216064808) ) ) {
                  if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)2.393745899200439897) ) ) {
                    result[0] += 0.08545318226881596;
                  } else {
                    result[0] += -0.10465373106726067;
                  }
                } else {
                  result[0] += -0.10720014382562132;
                }
              }
            }
          }
        }
      } else {
        result[0] += -0.034183413610141544;
      }
    }
  }
  if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
    if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)76.50000000000001421) ) ) {
      result[0] += 0.0003463426275175703;
    } else {
      if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)13.1100659370422381) ) ) {
        if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)83.50000000000001421) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.071567356586456743) ) ) {
            result[0] += 0.0399089484424378;
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.219399690628052646) ) ) {
              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.868834793567657693) ) ) {
                result[0] += -0.23780402427881764;
              } else {
                result[0] += -0.042936974373911724;
              }
            } else {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.028861761093140537) ) ) {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                  result[0] += -0.006501454048825387;
                } else {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.450390577316285068) ) ) {
                    result[0] += -0.07072161410889759;
                  } else {
                    result[0] += 0.0017566026069201605;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                  result[0] += -0.04285173455942871;
                } else {
                  result[0] += 0.0744396251570522;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.617236852645874912) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.743881702423096591) ) ) {
                    result[0] += 0.002380945497750182;
                  } else {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)3.449861526489258257) ) ) {
                        result[0] += -0.08966600640281244;
                      } else {
                        result[0] += -0.02050438151083142;
                      }
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.901921629905701128) ) ) {
                        result[0] += -0.0638747910724585;
                      } else {
                        result[0] += 0.016798925987306;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)221.5000000000000284) ) ) {
                    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.97887301445007413) ) ) {
                        if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                          result[0] += 0.03710616431375156;
                        } else {
                          result[0] += 0.010783392353526683;
                        }
                      } else {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.288152217864991123) ) ) {
                          result[0] += 0.017025735598492048;
                        } else {
                          result[0] += -0.02355844117477008;
                        }
                      }
                    } else {
                      result[0] += -0.02768867345510688;
                    }
                  } else {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.255632162094117099) ) ) {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
                        result[0] += -0.015946313167925354;
                      } else {
                        result[0] += -0.045785861013123534;
                      }
                    } else {
                      if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)4.182021141052246982) ) ) {
                        result[0] += 0.009434090877653034;
                      } else {
                        result[0] += -0.03409166580184503;
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.0030997255887751955;
                } else {
                  if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.831997871398927558) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.775349855422974521) ) ) {
                      result[0] += 0.014074284293941579;
                    } else {
                      if ( UNLIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)6.000000000000000888) ) ) {
                        result[0] += 0.03176896317834053;
                      } else {
                        result[0] += 0.0915052470975608;
                      }
                    }
                  } else {
                    result[0] += -0.05064968027661011;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.835998296737671787) ) ) {
                  result[0] += -0.0013368056803986371;
                } else {
                  if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.40000796318054288) ) ) {
                        result[0] += -0.05909164331315919;
                      } else {
                        if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.725620865821838823) ) ) {
                          result[0] += -0.021679000024978107;
                        } else {
                          if ( LIKELY( !(data[58].missing != -1) || (data[58].fvalue <= (double)1.500000000000000222) ) ) {
                            if ( LIKELY( !(data[56].missing != -1) || (data[56].fvalue <= (double)3.000000000000000444) ) ) {
                              result[0] += -0.09232933847538172;
                            } else {
                              result[0] += -0.0358490150568331;
                            }
                          } else {
                            result[0] += 0.009314720059156806;
                          }
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.288254261016846591) ) ) {
                        result[0] += -0.05368897132239686;
                      } else {
                        if ( UNLIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)24.00000000000000355) ) ) {
                          result[0] += -0.020770013533075726;
                        } else {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.587308406829834873) ) ) {
                            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)199.5000000000000284) ) ) {
                              result[0] += -0.05214779409750895;
                            } else {
                              result[0] += -0.0005787290022893886;
                            }
                          } else {
                            result[0] += 0.01639024642923209;
                          }
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.700598716735840066) ) ) {
                          result[0] += -0.03890501423875833;
                        } else {
                          result[0] += 0.03263693372752665;
                        }
                      } else {
                        if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += -0.07026428586305482;
                        } else {
                          result[0] += -0.004195518719360989;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.54081821441650568) ) ) {
                        result[0] += 0.0018290466914095858;
                      } else {
                        result[0] += -0.03531401170293572;
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.363266706466675693) ) ) {
                  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.97887301445007413) ) ) {
                      result[0] += 0.03425010386160626;
                    } else {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.467917680740357333) ) ) {
                        if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.95797300338745206) ) ) {
                          result[0] += 0.03341513595540595;
                        } else {
                          result[0] += -0.1090634605845632;
                        }
                      } else {
                        result[0] += -0.011838593933502166;
                      }
                    }
                  } else {
                    result[0] += 0.004027572222271933;
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.094205617904663974) ) ) {
                    result[0] += 0.048236999658806706;
                  } else {
                    if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                        result[0] += -0.011386408531018376;
                      } else {
                        result[0] += -0.06565200475132985;
                      }
                    } else {
                      if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)155.5000000000000284) ) ) {
                        if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += 0.034515192201213296;
                        } else {
                          result[0] += -0.02398473277637917;
                        }
                      } else {
                        if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.868834793567657693) ) ) {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.210240364074708808) ) ) {
                            result[0] += -0.01481924275423213;
                          } else {
                            result[0] += -0.04909461431729978;
                          }
                        } else {
                          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.53326439857482999) ) ) {
                            result[0] += 0.08294709738627642;
                          } else {
                            result[0] += -0.08052388877139871;
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          } else {
            result[0] += -0.019359030788632974;
          }
        }
      } else {
        if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
          result[0] += 0.034393047896961015;
        } else {
          result[0] += -0.05566693909966761;
        }
      }
    }
  } else {
    result[0] += 0.0007179513175194235;
  }
  if ( UNLIKELY(  (data[35].missing != -1) && (data[35].fvalue <= (double)-1.00000001800250948e-35) ) ) {
    result[0] += 0.08254676908536812;
  } else {
    if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)6.000000000000000888) ) ) {
      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.72660112380981623) ) ) {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
          if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.835998296737671787) ) ) {
                  if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.48298668861389249) ) ) {
                    if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.02783010961684961;
                    } else {
                      result[0] += 0.002271620503929026;
                    }
                  } else {
                    if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                      if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.722943305969239169) ) ) {
                        result[0] += 0.15407539798132214;
                      } else {
                        if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
                          result[0] += -0.07457789452629761;
                        } else {
                          result[0] += 0.08052745146629235;
                        }
                      }
                    } else {
                      result[0] += -0.0804718150031377;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.439304351806642401) ) ) {
                    result[0] += -0.014375507566260543;
                  } else {
                    if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += -0.0800921781066617;
                    } else {
                      result[0] += -0.0004216929587210855;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.605120182037354404) ) ) {
                    result[0] += 0.14348803444739847;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.596743106842042792) ) ) {
                      if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
                        result[0] += -0.07218153064488118;
                      } else {
                        if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.497866153717041238) ) ) {
                          result[0] += -0.06845031650647532;
                        } else {
                          if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.700598716735840066) ) ) {
                            result[0] += 0.10519622189487554;
                          } else {
                            result[0] += -0.033904449064669696;
                          }
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
                        result[0] += 0.04357223915512277;
                      } else {
                        result[0] += -0.0371884337785106;
                      }
                    }
                  }
                } else {
                  result[0] += 0.044686027746036194;
                }
              }
            } else {
              result[0] += 0.007045141936011615;
            }
          } else {
            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.449861526489258257) ) ) {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.700598716735840066) ) ) {
                      result[0] += 0.0033156841588712427;
                    } else {
                      result[0] += -0.04578625440194048;
                    }
                  } else {
                    result[0] += -0.012302348352965366;
                  }
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.944155693054200995) ) ) {
                    result[0] += -0.022313536033993;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.129780292510988104) ) ) {
                      result[0] += 0.1236758835248629;
                    } else {
                      result[0] += 0.028649128636501892;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.288254261016846591) ) ) {
                    if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += 0.04908325200223653;
                    } else {
                      result[0] += 0.005371139972648827;
                    }
                  } else {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)5.500000000000000888) ) ) {
                      result[0] += 0.005825569505946126;
                    } else {
                      result[0] += -0.04451409759647862;
                    }
                  }
                } else {
                  result[0] += 0.05532252057929641;
                }
              }
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.835998296737671787) ) ) {
                  if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += -0.010155059593064683;
                  } else {
                    if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)9.113774299621583808) ) ) {
                      if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)20.50000000000000355) ) ) {
                        result[0] += -0.08661427645496428;
                      } else {
                        if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.83939445018768355) ) ) {
                          result[0] += -0.038569378705366565;
                        } else {
                          result[0] += 0.0362540179750465;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += 0.09879218091592029;
                      } else {
                        result[0] += 0.022415235323283718;
                      }
                    }
                  }
                } else {
                  result[0] += -0.0415382166447378;
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.397998809814454013) ) ) {
                  result[0] += -0.006067379380500502;
                } else {
                  if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.701225757598877397) ) ) {
                    result[0] += -0.024016919019374944;
                  } else {
                    result[0] += -0.08614754771135959;
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
            result[0] += -0.0403935209616142;
          } else {
            if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += -0.030280477682486456;
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.873467922210695136) ) ) {
                if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.384246587753296343) ) ) {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
                      if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += 0.0519996219992701;
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.851041555404663974) ) ) {
                          result[0] += 0.040596097299278384;
                        } else {
                          result[0] += -0.02039828510224423;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.590985536575318271) ) ) {
                        result[0] += -0.08242358753512578;
                      } else {
                        result[0] += -0.010197060505263249;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
                      result[0] += 0.015172247054580324;
                    } else {
                      result[0] += -0.045697645558200106;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)6.500000000000000888) ) ) {
                    result[0] += 0.029702590363467014;
                  } else {
                    result[0] += -0.06140653969621815;
                  }
                }
              } else {
                result[0] += -0.022885207148680392;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
          result[0] += 0.0037586643741474406;
        } else {
          if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.791641235351563388) ) ) {
            if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += -0.07944753723981589;
            } else {
              if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += -0.05021758009278546;
              } else {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.042985591845727894;
                } else {
                  result[0] += -0.057588372189135865;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
              result[0] += -0.013915370263861322;
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += -0.04685380018507171;
                } else {
                  result[0] += -0.1059573781436407;
                }
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.553655147552491123) ) ) {
                  result[0] += -0.0499039318706298;
                } else {
                  result[0] += 0.050032373965296785;
                }
              }
            }
          }
        }
      }
    } else {
      result[0] += 0.00014026657556194395;
    }
  }
}

