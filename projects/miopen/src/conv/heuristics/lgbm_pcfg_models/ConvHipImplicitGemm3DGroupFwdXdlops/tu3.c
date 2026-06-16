
#include "header.h"

void predict_unit3(union Entry* data, double* result) {
  unsigned int tmp;
  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
      result[0] += 0.06438884363600941;
    } else {
      if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.605039834976196733) ) ) {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.154959201812744585) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)0.8958797454833985485) ) ) {
            result[0] += 0.0009396638790592934;
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.700598716735840066) ) ) {
              result[0] += 0.11908045909678043;
            } else {
              if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.004485678388785474;
              } else {
                result[0] += 0.08892960858704058;
              }
            }
          }
        } else {
          result[0] += 0.0029205684291511245;
        }
      } else {
        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
            result[0] += 0.013870526904391845;
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.95211219787597834) ) ) {
              result[0] += 0.07682083326737293;
            } else {
              result[0] += -0.03717322599567815;
            }
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.827801465988160068) ) ) {
            result[0] += -0.07096949653020893;
          } else {
            result[0] += 0.01342360024382387;
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
      if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
              result[0] += -0.0031198818981898017;
            } else {
              result[0] += -0.09975822489436334;
            }
          } else {
            result[0] += 0.0005325823960388119;
          }
        } else {
          if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += -0.10040136162537344;
          } else {
            if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.13168312864808074;
            } else {
              result[0] += -0.017356051708511063;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.342454433441162998) ) ) {
                    if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.029068946838379794) ) ) {
                        result[0] += 0.08315370374459437;
                      } else {
                        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                          result[0] += -0.08168692141804024;
                        } else {
                          result[0] += 0.19785885033979936;
                        }
                      }
                    } else {
                      result[0] += -0.02610658831958544;
                    }
                  } else {
                    result[0] += -0.04304957106452104;
                  }
                } else {
                  if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                    if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += -0.04557594682877302;
                    } else {
                      if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                        result[0] += -0.05721432305338541;
                      } else {
                        result[0] += 0.0424337539005265;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.918272972106934482) ) ) {
                      result[0] += -0.09207160619275508;
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
                        result[0] += 0.06984559142260229;
                      } else {
                        result[0] += -0.036140037052958596;
                      }
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.004644162169370488;
                  } else {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += 0.017521476481056893;
                    } else {
                      if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += 0.11424417909800624;
                      } else {
                        result[0] += 0.005571589123462885;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.029068946838379794) ) ) {
                    if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                      result[0] += 0.0736966797817673;
                    } else {
                      result[0] += -0.009326324133113807;
                    }
                  } else {
                    result[0] += 0.019987268443507705;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.302512168884278232) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.172047138214112216) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.284418344497681552) ) ) {
                    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                      result[0] += 0.0715000924006348;
                    } else {
                      result[0] += -0.10164402703230234;
                    }
                  } else {
                    result[0] += 0.12482460686891306;
                  }
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.56941866874694913) ) ) {
                    if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.09113749536476927;
                    } else {
                      result[0] += 0.02613835965008058;
                    }
                  } else {
                    result[0] += 0.03119801174901063;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                  result[0] += -0.000290039366574375;
                } else {
                  result[0] += 0.1023295009402615;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += 0.004706319123595506;
              } else {
                result[0] += -0.10848957146562244;
              }
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.607751369476319248) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.142747402191162998) ) ) {
                  result[0] += -0.04088704433224936;
                } else {
                  result[0] += 0.045892477669904355;
                }
              } else {
                result[0] += 0.07300854295557155;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.357691764831543413) ) ) {
              result[0] += -0.01022135624580954;
            } else {
              result[0] += -0.0780293404941218;
            }
          } else {
            result[0] += 0.0005847833002655616;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)1.497866153717041238) ) ) {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)15.02900028228759943) ) ) {
          result[0] += -0.029893043340696115;
        } else {
          result[0] += -0.1843457533815307;
        }
      } else {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
          if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)6.000000000000000888) ) ) {
              if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.08079919331770813;
                } else {
                  result[0] += 0.012269676448574846;
                }
              } else {
                result[0] += -0.06147017033508914;
              }
            } else {
              result[0] += -0.002496916471729734;
            }
          } else {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
                result[0] += 0.07288546949135859;
              } else {
                result[0] += -0.030584753361435263;
              }
            } else {
              result[0] += 0.030052877452469845;
            }
          }
        } else {
          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += 0.08280479932834625;
            } else {
              result[0] += -0.0348293563157997;
            }
          } else {
            result[0] += -0.08667785167415465;
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
      result[0] += 0.06109458802869795;
    } else {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.154959201812744585) ) ) {
        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
          if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += 0.019838532520759927;
          } else {
            result[0] += 0.0710861008372185;
          }
        } else {
          result[0] += -0.033441302135889354;
        }
      } else {
        if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.760117292404175249) ) ) {
          result[0] += 0.0028686682201305397;
        } else {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
              if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
                if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)21466447872.00000381) ) ) {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.597218394279480425) ) ) {
                    result[0] += 0.023247889451240525;
                  } else {
                    result[0] += -0.06546086233991412;
                  }
                } else {
                  if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)137422176256.0000153) ) ) {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.31402075290679976) ) ) {
                      result[0] += -0.011908546401377298;
                    } else {
                      result[0] += -0.12197880727197913;
                    }
                  } else {
                    result[0] += 0.006820290638134336;
                  }
                }
              } else {
                result[0] += 0.07431251861316283;
              }
            } else {
              result[0] += 0.06569771735173371;
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.827801465988160068) ) ) {
              result[0] += -0.07346874218511024;
            } else {
              result[0] += 0.01291567091322744;
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
      if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2252.000000000000455) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.215408444404602495) ) ) {
          result[0] += 0.005119493274078052;
        } else {
          if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += -0.07282621945249258;
          } else {
            result[0] += -0.02146451035197122;
          }
        }
      } else {
        result[0] += -0.0019490289044862754;
      }
    } else {
      if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)4.500000000000000888) ) ) {
        result[0] += 0.002468653586829366;
      } else {
        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.909855604171753818) ) ) {
            if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)8816427008.000001907) ) ) {
              result[0] += -0.052813348759774836;
            } else {
              if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)2.087193608283997026) ) ) {
                if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.94957673549652144) ) ) {
                      result[0] += 0.00478024874559018;
                    } else {
                      result[0] += 0.1259875588057526;
                    }
                  } else {
                    result[0] += -0.025159308627143353;
                  }
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.827801465988160068) ) ) {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.778982400894165927) ) ) {
                        result[0] += 0.02307885647701491;
                      } else {
                        result[0] += -0.05834903378501397;
                      }
                    } else {
                      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.607751369476319248) ) ) {
                        result[0] += -0.012330365571241004;
                      } else {
                        result[0] += 0.09606518336969248;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.743881702423096591) ) ) {
                          result[0] += 0.0009604439014921184;
                        } else {
                          result[0] += -0.05305267405418839;
                        }
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.778982400894165927) ) ) {
                          result[0] += 0.10378735514618204;
                        } else {
                          result[0] += 0.008829772870795734;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.172047138214112216) ) ) {
                          result[0] += 0.012665714803767082;
                        } else {
                          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                            result[0] += 0.030811655220689644;
                          } else {
                            result[0] += 0.08519128609974011;
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.198464870452881303) ) ) {
                          result[0] += -0.040043207057614015;
                        } else {
                          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.607751369476319248) ) ) {
                            result[0] += 0.004120823478103142;
                          } else {
                            result[0] += 0.09379574750933542;
                          }
                        }
                      }
                    }
                  }
                }
              } else {
                result[0] += -0.11895909998497396;
              }
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.467917680740357333) ) ) {
              if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                result[0] += 0.027413442552507924;
              } else {
                result[0] += -0.05158790751751552;
              }
            } else {
              result[0] += -0.051558908602317134;
            }
          }
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.909855604171753818) ) ) {
            if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.651049375534058505) ) ) {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.313157320022583896) ) ) {
                    if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.108761310577394354) ) ) {
                      result[0] += -0.0015665786040351126;
                    } else {
                      result[0] += -0.05994261242620019;
                    }
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.484580039978028232) ) ) {
                      if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)9.500000000000001776) ) ) {
                        result[0] += -0.08540023347275569;
                      } else {
                        result[0] += -0.00032494068089481055;
                      }
                    } else {
                      result[0] += 0.01819257879199899;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)208.0000000000000284) ) ) {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += 0.033593717742946255;
                    } else {
                      result[0] += -0.05329270792138518;
                    }
                  } else {
                    result[0] += -0.09334695315597921;
                  }
                }
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.302512168884278232) ) ) {
                  result[0] += -0.06741361035184981;
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.607751369476319248) ) ) {
                    if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                      result[0] += 0.027096444298981566;
                    } else {
                      result[0] += -0.12134480656534188;
                    }
                  } else {
                    result[0] += -0.04161765624524265;
                  }
                }
              }
            } else {
              result[0] += -0.0029241358797127467;
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.029068946838379794) ) ) {
              if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)3072.000000000000455) ) ) {
                result[0] += 0.03719295735396088;
              } else {
                if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.03908157599040944;
                } else {
                  result[0] += -0.04465947017833222;
                }
              }
            } else {
              if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
                  if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.05442971817753272;
                  } else {
                    result[0] += 0.01105965624522091;
                  }
                } else {
                  result[0] += 0.10806673461834831;
                }
              } else {
                if ( UNLIKELY(  (data[43].missing != -1) && (data[43].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += 0.06343038347902284;
                  } else {
                    result[0] += -0.0019475241037161398;
                  }
                } else {
                  result[0] += -0.09770335357026423;
                }
              }
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2727.500000000000455) ) ) {
    if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)10.50000000000000178) ) ) {
      if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
        result[0] += 0.0018261172271680911;
      } else {
        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.172047138214112216) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.94957673549652144) ) ) {
                if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.715336322784424716) ) ) {
                    result[0] += 0.009303473763569247;
                  } else {
                    result[0] += -0.10667317406682267;
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.41211462020874201) ) ) {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.524927973747253862) ) ) {
                      result[0] += -0.030164114283766637;
                    } else {
                      result[0] += -0.15284712163326042;
                    }
                  } else {
                    result[0] += -0.013040385445471293;
                  }
                }
              } else {
                result[0] += 0.01260218598511871;
              }
            } else {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)46.00000000000000711) ) ) {
                if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.06739160801176251;
                } else {
                  result[0] += -0.05034070412953462;
                }
              } else {
                result[0] += -0.006606699268757896;
              }
            }
          } else {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.450390577316285068) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
                result[0] += -0.005733345804634126;
              } else {
                result[0] += -0.027357745934560786;
              }
            } else {
              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.431901693344116655) ) ) {
                result[0] += 0.13429237692424;
              } else {
                result[0] += -0.04803553258982818;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
              result[0] += -0.0009885294724390667;
            } else {
              if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                result[0] += -0.02466222639408165;
              } else {
                result[0] += -0.07209104780063706;
              }
            }
          } else {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
              result[0] += -0.028997262528775603;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.108761310577394354) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.208071470260621005) ) ) {
                  result[0] += -0.07086040807633907;
                } else {
                  result[0] += 0.05195743515598242;
                }
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.142630577087403232) ) ) {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                      result[0] += 0.12115765768850438;
                    } else {
                      result[0] += -0.015931133318809763;
                    }
                  } else {
                    result[0] += -0.08314762181851873;
                  }
                } else {
                  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                        result[0] += -0.11456826796228102;
                      } else {
                        result[0] += 0.011543399755827707;
                      }
                    } else {
                      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.607751369476319248) ) ) {
                          if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                            result[0] += -0.05350073857194008;
                          } else {
                            result[0] += 0.004725141606293903;
                          }
                        } else {
                          result[0] += 0.007312546592394656;
                        }
                      } else {
                        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
                          result[0] += -0.004441402515388074;
                        } else {
                          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                            result[0] += 0.05034001682416836;
                          } else {
                            result[0] += -0.0002758532494279706;
                          }
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                      if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += 0.006279288441038641;
                      } else {
                        result[0] += 0.026270306201073557;
                      }
                    } else {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.607751369476319248) ) ) {
                        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)6.000000000000000888) ) ) {
                          result[0] += 0.11933227799189311;
                        } else {
                          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.302512168884278232) ) ) {
                            result[0] += -0.012733202070097503;
                          } else {
                            result[0] += 0.09096296992931523;
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.993164777755738193) ) ) {
                          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.715336322784424716) ) ) {
                            result[0] += -0.03802351755988727;
                          } else {
                            result[0] += -0.1499634724058508;
                          }
                        } else {
                          result[0] += 0.04787256332291576;
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
      result[0] += -0.029225120981767095;
    }
  } else {
    if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.700598716735840066) ) ) {
        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
          result[0] += -0.01398390498835204;
        } else {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
            result[0] += 0.08666360861698155;
          } else {
            result[0] += -0.00440243442201655;
          }
        }
      } else {
        if ( UNLIKELY(  (data[27].missing != -1) && (data[27].fvalue <= (double)-1.00000001800250948e-35) ) ) {
          result[0] += 0.08008154315599886;
        } else {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.651049375534058505) ) ) {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.516936540603638583) ) ) {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += 0.09672045346556787;
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.45958471298217951) ) ) {
                      result[0] += 0.0037605211904590563;
                    } else {
                      result[0] += 0.08563786094618916;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.030390171709750027;
                  } else {
                    result[0] += 0.0364412665533034;
                  }
                }
              } else {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += -0.13949176610016503;
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.802696108818054643) ) ) {
                    result[0] += 0.15295061704695045;
                  } else {
                    result[0] += -0.10904764578884876;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.970085620880127397) ) ) {
                result[0] += 0.0665537010013257;
              } else {
                result[0] += -0.06270302425959559;
              }
            }
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.802696108818054643) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.516936540603638583) ) ) {
                result[0] += -0.044792936120287834;
              } else {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += 0.06163575511213846;
                } else {
                  result[0] += -0.05855933814356215;
                }
              }
            } else {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += -0.0198262724226354;
              } else {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += 0.06723535844913313;
                  } else {
                    result[0] += -0.030703810747490592;
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.456172943115236151) ) ) {
                    result[0] += -0.04130421097205481;
                  } else {
                    result[0] += 0.02058892622413338;
                  }
                }
              }
            }
          }
        }
      }
    } else {
      result[0] += -0.02458275395610443;
    }
  }
  if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2727.500000000000455) ) ) {
    if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)10.50000000000000178) ) ) {
      if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.90474271774292081) ) ) {
          result[0] += 0.0011916463393029308;
        } else {
          if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)4.500000000000000888) ) ) {
            result[0] += 0.01913486174726206;
          } else {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)46.00000000000000711) ) ) {
              result[0] += 0.07918059148644063;
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.947818994522095615) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.172047138214112216) ) ) {
                    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += 0.7149076504106002;
                    } else {
                      result[0] += 0.19685785079435386;
                    }
                  } else {
                    result[0] += 0.05831090886971423;
                  }
                } else {
                  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)8.500000000000001776) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.327068090438843662) ) ) {
                      result[0] += -0.08125675774209679;
                    } else {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.347890853881836826) ) ) {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.172047138214112216) ) ) {
                          result[0] += 0.3509679160224893;
                        } else {
                          result[0] += -0.008882911240807343;
                        }
                      } else {
                        result[0] += -0.030354817741480473;
                      }
                    }
                  } else {
                    result[0] += 0.04959548961908839;
                  }
                }
              } else {
                result[0] += -0.06712817890290429;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.172047138214112216) ) ) {
            if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.94957673549652144) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.715336322784424716) ) ) {
                  if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += 0.016892890814360236;
                  } else {
                    result[0] += -0.01387147086201026;
                  }
                } else {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += 0.059527272586102464;
                  } else {
                    result[0] += -0.1377089116301062;
                  }
                }
              } else {
                result[0] += 0.010705734660835695;
              }
            } else {
              if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.357691764831543413) ) ) {
                  if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.01770896159789105;
                  } else {
                    if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += 0.030127029153678216;
                    } else {
                      result[0] += -0.09176840436745766;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)6.000000000000000888) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.807813167572023261) ) ) {
                      result[0] += 0.03268657595243099;
                    } else {
                      result[0] += -0.004624019584842528;
                    }
                  } else {
                    result[0] += -0.02885964361072966;
                  }
                }
              } else {
                result[0] += -0.013454151287534535;
              }
            }
          } else {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.450390577316285068) ) ) {
              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.008153491502309205;
                } else {
                  result[0] += 0.03191452444809431;
                }
              } else {
                result[0] += -0.026262247884295972;
              }
            } else {
              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.431901693344116655) ) ) {
                result[0] += 0.12102963295711378;
              } else {
                result[0] += -0.045868843904759156;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
              result[0] += -0.0009794304508291983;
            } else {
              if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                result[0] += -0.02211183315941917;
              } else {
                result[0] += -0.06551778901246223;
              }
            }
          } else {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.90474271774292081) ) ) {
                result[0] += -0.02030002920664077;
              } else {
                result[0] += -0.08136757350522576;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.108761310577394354) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.208071470260621005) ) ) {
                  result[0] += -0.06554641753563457;
                } else {
                  result[0] += 0.04652317164173302;
                }
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.142630577087403232) ) ) {
                  result[0] += -0.014667619639775179;
                } else {
                  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                        result[0] += -0.11328792240907551;
                      } else {
                        result[0] += 0.01529014127715326;
                      }
                    } else {
                      if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                        result[0] += 0.0014877834703853974;
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.397998809814454013) ) ) {
                          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                            result[0] += 0.11031342499711856;
                          } else {
                            result[0] += 0.015560093199049788;
                          }
                        } else {
                          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                            result[0] += -0.026900047117560045;
                          } else {
                            result[0] += 0.03651197244355994;
                          }
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                      if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += 0.004994920511584754;
                      } else {
                        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.484580039978028232) ) ) {
                          if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)6.000000000000000888) ) ) {
                            result[0] += 0.008930239126025237;
                          } else {
                            result[0] += -0.09508298514204819;
                          }
                        } else {
                          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)46.00000000000000711) ) ) {
                            result[0] += -0.051153704203434985;
                          } else {
                            if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                              result[0] += 0.07284721395140056;
                            } else {
                              result[0] += 0.021544121842543444;
                            }
                          }
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.607751369476319248) ) ) {
                        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)6.000000000000000888) ) ) {
                          result[0] += 0.11411495895304143;
                        } else {
                          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                            result[0] += -0.008252104985307948;
                          } else {
                            result[0] += 0.09537513569862306;
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.993164777755738193) ) ) {
                          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.851041555404663974) ) ) {
                            result[0] += -0.036679504135952716;
                          } else {
                            result[0] += -0.13972627786672445;
                          }
                        } else {
                          result[0] += 0.03865541727398006;
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
      result[0] += -0.026130299759778222;
    }
  } else {
    if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.418141007423401323) ) ) {
        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
          result[0] += -0.013989277194700951;
        } else {
          if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)12.00000000000000178) ) ) {
            result[0] += 0.08140981529724933;
          } else {
            result[0] += -0.008821862622180354;
          }
        }
      } else {
        if ( UNLIKELY(  (data[27].missing != -1) && (data[27].fvalue <= (double)-1.00000001800250948e-35) ) ) {
          result[0] += 0.07764886469879473;
        } else {
          result[0] += 0.006917508406388726;
        }
      }
    } else {
      result[0] += -0.023996394476025067;
    }
  }
  if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
    if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)40.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
              if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += 0.01118088803547721;
              } else {
                result[0] += -0.06098647331165518;
              }
            } else {
              result[0] += 0.06207743266716564;
            }
          } else {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += 0.14428420426650393;
              } else {
                result[0] += 0.05908055172919278;
              }
            } else {
              result[0] += 0.03402795892164281;
            }
          }
        } else {
          if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
            result[0] += -0.06358068491500353;
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.699081301689148393) ) ) {
              result[0] += 0.021736753521681817;
            } else {
              result[0] += -0.03280796123877488;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)257681260544.0000305) ) ) {
                  result[0] += -0.0064828776432833;
                } else {
                  result[0] += 0.02332228905539935;
                }
              } else {
                result[0] += 0.00738259554683829;
              }
            } else {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.0229394609938399;
              } else {
                if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.770631790161133257) ) ) {
                  result[0] += 0.051221968048170356;
                } else {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.484580039978028232) ) ) {
                      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                        result[0] += 0.06649081557321976;
                      } else {
                        result[0] += 0.007445675489274155;
                      }
                    } else {
                      result[0] += -0.006339447397208892;
                    }
                  } else {
                    if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                      result[0] += -0.024282434395197616;
                    } else {
                      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                        result[0] += -0.027404739899636807;
                      } else {
                        result[0] += 0.007213401077679856;
                      }
                    }
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.45958471298217951) ) ) {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.06402193426316531;
                } else {
                  result[0] += 0.0005607466847794548;
                }
              } else {
                result[0] += -0.049902645943891294;
              }
            } else {
              result[0] += 0.030557460597535724;
            }
          }
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.141444921493531162) ) ) {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += -0.05353919622883327;
            } else {
              if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.651049375534058505) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.484580039978028232) ) ) {
                      if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += 0.01717887022726715;
                      } else {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.208071470260621005) ) ) {
                          result[0] += -0.03808325682322758;
                        } else {
                          result[0] += 0.07942075512485572;
                        }
                      }
                    } else {
                      result[0] += 0.06496021415322718;
                    }
                  } else {
                    if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                      if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                        result[0] += -0.1133523088535591;
                      } else {
                        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
                          result[0] += 0.025521152315245807;
                        } else {
                          result[0] += -0.05632435601444535;
                        }
                      }
                    } else {
                      result[0] += 0.004188440665958524;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.302512168884278232) ) ) {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                          result[0] += 0.005678674851025501;
                        } else {
                          result[0] += 0.10311336374963949;
                        }
                      } else {
                        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.067782521247864214) ) ) {
                          result[0] += 0.0004614865275272251;
                        } else {
                          if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.172047138214112216) ) ) {
                              result[0] += 0.15578897355864715;
                            } else {
                              result[0] += 0.009804406395632126;
                            }
                          } else {
                            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.349750161170959917) ) ) {
                              result[0] += 0.1477708077270665;
                            } else {
                              result[0] += 0.04833212823147145;
                            }
                          }
                        }
                      }
                    } else {
                      result[0] += -0.028548972669357388;
                    }
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.607751369476319248) ) ) {
                      if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                        result[0] += -0.07450883335895658;
                      } else {
                        result[0] += 0.004548504783008428;
                      }
                    } else {
                      if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                          result[0] += -0.02512355101509688;
                        } else {
                          result[0] += 0.06577597870282328;
                        }
                      } else {
                        result[0] += 0.0300063963382271;
                      }
                    }
                  }
                }
              } else {
                result[0] += 0.04233112535517947;
              }
            }
          } else {
            result[0] += -0.033548053883853825;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
        if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)2.500000000000000444) ) ) {
          if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += -0.01973752107071433;
          } else {
            result[0] += 0.023232099348288834;
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.484580039978028232) ) ) {
                result[0] += -0.002896042675580912;
              } else {
                result[0] += -0.0553280434732337;
              }
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.778982400894165927) ) ) {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                  result[0] += 0.03425574517771889;
                } else {
                  result[0] += -0.05584657510062024;
                }
              } else {
                result[0] += 0.039031046132782715;
              }
            }
          } else {
            result[0] += 0.016845031538239612;
          }
        }
      } else {
        if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.38936424255371271) ) ) {
            result[0] += 0.11467195373945467;
          } else {
            result[0] += -0.1387922778189409;
          }
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.585059762001038042) ) ) {
            result[0] += -0.019056288291339263;
          } else {
            result[0] += -0.12024035662309102;
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.552972793579102007) ) ) {
      result[0] += -0.0869329282716721;
    } else {
      if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
        result[0] += -0.00950147177847907;
      } else {
        result[0] += 0.0026450976040060547;
      }
    }
  }
  if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2727.500000000000455) ) ) {
    if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)10.50000000000000178) ) ) {
      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
        if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += 0.0005940758928463692;
        } else {
          if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += -0.0015353299581637543;
            } else {
              result[0] += -0.02145575840602018;
            }
          } else {
            result[0] += -0.02463177300153063;
          }
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.357691764831543413) ) ) {
          result[0] += -0.025331907260816147;
        } else {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.142630577087403232) ) ) {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.48918962478637873) ) ) {
                result[0] += 0.02848142302310591;
              } else {
                result[0] += -0.0055575672833342174;
              }
            } else {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += -0.048697049545683865;
              } else {
                if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2252.000000000000455) ) ) {
                  result[0] += -0.06505571311195409;
                } else {
                  result[0] += 0.10620960952205954;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.311204195022583896) ) ) {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.028458572428893665;
                } else {
                  result[0] += 0.016869114648434966;
                }
              } else {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.379217386245728427) ) ) {
                    if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.007498628725783023;
                    } else {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.255632162094117099) ) ) {
                        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
                          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.284418344497681552) ) ) {
                            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)40.00000000000000711) ) ) {
                              result[0] += 0.14372316179797562;
                            } else {
                              result[0] += -0.010556066315981263;
                            }
                          } else {
                            result[0] += 0.08842771165513685;
                          }
                        } else {
                          result[0] += -0.006391382114340784;
                        }
                      } else {
                        result[0] += -0.0025297434281544907;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.90638875961303889) ) ) {
                        result[0] += 0.0011325590837298432;
                      } else {
                        result[0] += -0.040817964430064975;
                      }
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.607751369476319248) ) ) {
                        result[0] += 0.06034359815414767;
                      } else {
                        result[0] += -0.00021598131612901364;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.516936540603638583) ) ) {
                      result[0] += -0.008614683249981114;
                    } else {
                      result[0] += 0.0070944424376287426;
                    }
                  } else {
                    result[0] += -0.033231102489698364;
                  }
                }
              }
            } else {
              result[0] += 0.027203135141114606;
            }
          }
        }
      }
    } else {
      result[0] += -0.02585648103757764;
    }
  } else {
    if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
      if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
        if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.870983839035034624) ) ) {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.481121778488159624) ) ) {
              result[0] += -0.022456257720696215;
            } else {
              result[0] += -0.0036511774332208344;
            }
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.142630577087403232) ) ) {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.10675222735586086;
                } else {
                  result[0] += 0.030517571551588532;
                }
              } else {
                result[0] += 0.12219715241172019;
              }
            } else {
              result[0] += -0.07050357333785089;
            }
          }
        } else {
          if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += 0.03533848924579409;
          } else {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += 0.04125932516958952;
            } else {
              result[0] += -0.14196463355749547;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)10.50000000000000178) ) ) {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.142630577087403232) ) ) {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
              result[0] += 0.03298677531153293;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.842307567596437323) ) ) {
                result[0] += 0.02046120791802432;
              } else {
                result[0] += -0.026039360371140075;
              }
            }
          } else {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.90638875961303889) ) ) {
                result[0] += 0.043545187492355406;
              } else {
                result[0] += -0.06650939053563962;
              }
            } else {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.028695401784530894;
                } else {
                  result[0] += 0.08684708378437293;
                }
              } else {
                result[0] += 0.013997937559767533;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.736160039901734287) ) ) {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += 0.11392696970600645;
            } else {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.238486170768738237) ) ) {
                result[0] += 0.04896360625535719;
              } else {
                result[0] += -0.10196295067841694;
              }
            }
          } else {
            result[0] += 0.09483871130957464;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.780340790748596635) ) ) {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)1.242453336715698464) ) ) {
          result[0] += 0.18806059729449964;
        } else {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.484580039978028232) ) ) {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.516936540603638583) ) ) {
                if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.050772342930022155;
                } else {
                  if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.76158928871154874) ) ) {
                    result[0] += 0.09893055544073293;
                  } else {
                    result[0] += -0.00741038167183438;
                  }
                }
              } else {
                result[0] += 0.1889933254225657;
              }
            } else {
              result[0] += -0.058198329522667985;
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.361115694046021396) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.029068946838379794) ) ) {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)208.0000000000000284) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.255632162094117099) ) ) {
                    result[0] += -0.018673406727006316;
                  } else {
                    if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += -0.09105594060180397;
                    } else {
                      result[0] += 0.08437690233160669;
                    }
                  }
                } else {
                  result[0] += -0.09947541526714875;
                }
              } else {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                  result[0] += -0.11138740896311233;
                } else {
                  result[0] += -0.007981602355163982;
                }
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.607751369476319248) ) ) {
                result[0] += 0.05639376788127773;
              } else {
                result[0] += -0.011366853233479112;
              }
            }
          }
        }
      } else {
        result[0] += -0.07373312442411606;
      }
    }
  }
  if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2727.500000000000455) ) ) {
    if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
      result[0] += 0.001220319060530547;
    } else {
      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
          if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += -0.01363703578213234;
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.993164777755738193) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
                  result[0] += -0.045869134205622164;
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.909855604171753818) ) ) {
                    result[0] += 0.006325218325839344;
                  } else {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                      result[0] += 0.04275286226249047;
                    } else {
                      result[0] += -0.11173710048101032;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)46.00000000000000711) ) ) {
                  result[0] += 0.1606714016182388;
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.255632162094117099) ) ) {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
                      if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += 0.12339425152172201;
                      } else {
                        result[0] += 0.04104651733667733;
                      }
                    } else {
                      result[0] += -0.008112452645971355;
                    }
                  } else {
                    result[0] += -0.022301618165315513;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.339395284652710849) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.849175214767456943) ) ) {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.042435407638550693) ) ) {
                    if ( LIKELY(  (data[42].missing != -1) && (data[42].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                      result[0] += 0.08318408597411923;
                    } else {
                      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.673553824424744096) ) ) {
                        result[0] += -0.04069283372618348;
                      } else {
                        result[0] += 0.06892026732204638;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += 0.01944944802792003;
                    } else {
                      result[0] += -0.07314252379561909;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
                    result[0] += -0.07483243290856532;
                  } else {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += -0.004753633742816622;
                    } else {
                      result[0] += 0.037052951930222254;
                    }
                  }
                }
              } else {
                result[0] += 0.056132374920599096;
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.172047138214112216) ) ) {
                result[0] += 0.009126873991004677;
              } else {
                result[0] += -0.02339502602804377;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.801661729812622958) ) ) {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.142630577087403232) ) ) {
                    result[0] += 0.04855892201612035;
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.484580039978028232) ) ) {
                      if ( UNLIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += -0.22453190960932662;
                      } else {
                        result[0] += 0.003353029930077419;
                      }
                    } else {
                      result[0] += -0.03386276127384626;
                    }
                  }
                } else {
                  result[0] += 0.015494523912174919;
                }
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.172047138214112216) ) ) {
                  if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                      if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                        result[0] += -0.02061955838718599;
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
                          result[0] += -0.05978895558266709;
                        } else {
                          result[0] += 0.03440003021784901;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                        result[0] += 0.061578932655974576;
                      } else {
                        result[0] += 0.021886138321228148;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
                      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.740319490432739702) ) ) {
                          result[0] += 0.01191117887865635;
                        } else {
                          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.198464870452881303) ) ) {
                            result[0] += -0.08438957623642176;
                          } else {
                            result[0] += -0.017465563630394223;
                          }
                        }
                      } else {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.718933820724488193) ) ) {
                          result[0] += 0.042006577396094806;
                        } else {
                          result[0] += -0.04817544930111225;
                        }
                      }
                    } else {
                      result[0] += -0.13241943712732415;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                    result[0] += -0.00015191423536922415;
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.302512168884278232) ) ) {
                      result[0] += 0.07160349629545844;
                    } else {
                      if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += 0.04694549822237093;
                      } else {
                        result[0] += -0.023483455090460598;
                      }
                    }
                  }
                }
              }
            } else {
              result[0] += -0.02806245653492222;
            }
          } else {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.743881702423096591) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.42478513717651456) ) ) {
                if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.205872535705568183) ) ) {
                    result[0] += 0.03165597261966654;
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.778982400894165927) ) ) {
                      result[0] += -0.02038796953896231;
                    } else {
                      if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += -0.007206877229086667;
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.029068946838379794) ) ) {
                          result[0] += 0.08308596340366547;
                        } else {
                          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                            result[0] += 0.14233319977538525;
                          } else {
                            result[0] += -0.003441902244575265;
                          }
                        }
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)9.500000000000001776) ) ) {
                    result[0] += -0.028492411932454515;
                  } else {
                    result[0] += 0.03270959519087322;
                  }
                }
              } else {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)208.0000000000000284) ) ) {
                  result[0] += 0.026047640768386185;
                } else {
                  result[0] += -0.05472581194461445;
                }
              }
            } else {
              result[0] += 0.03184766976022479;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)9.500000000000001776) ) ) {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.802901029586792436) ) ) {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
              result[0] += 0.0575145529889327;
            } else {
              result[0] += -0.014713348109330485;
            }
          } else {
            if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.87008237838745206) ) ) {
              result[0] += -0.037467765451358374;
            } else {
              result[0] += -0.1568715395421283;
            }
          }
        } else {
          result[0] += 0.004687797756057083;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.700598716735840066) ) ) {
        result[0] += -0.011505849085986952;
      } else {
        if ( UNLIKELY(  (data[27].missing != -1) && (data[27].fvalue <= (double)-1.00000001800250948e-35) ) ) {
          result[0] += 0.07509819636838391;
        } else {
          result[0] += 0.005175591454090702;
        }
      }
    } else {
      result[0] += -0.01962057309105035;
    }
  }
  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)0.8958797454833985485) ) ) {
    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)0.8958797454833985485) ) ) {
      result[0] += -0.08148450019013585;
    } else {
      result[0] += -0.0008539591987607796;
    }
  } else {
    if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)1.497866153717041238) ) ) {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)15.26177501678466975) ) ) {
          result[0] += 0.009014831205068555;
        } else {
          if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += 0.19389396399526673;
          } else {
            result[0] += 0.013991327576476857;
          }
        }
      } else {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
          if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
              if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.0728327634449956;
                } else {
                  result[0] += -0.023181816690922827;
                }
              } else {
                result[0] += 0.0012687193577637158;
              }
            } else {
              if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                  result[0] += -0.0018160888398751447;
                } else {
                  result[0] += -0.10517714653803806;
                }
              } else {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.442090511322023261) ) ) {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.08155846595764249) ) ) {
                      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
                        result[0] += 0.04262383168296857;
                      } else {
                        result[0] += -0.02859687872323486;
                      }
                    } else {
                      if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                        result[0] += -0.02886089927810075;
                      } else {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.736135363578796831) ) ) {
                          result[0] += 0.09275690536813716;
                        } else {
                          if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                            result[0] += -0.0672960487959918;
                          } else {
                            result[0] += 0.014712220825878409;
                          }
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.010044249342474219;
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.77496147155761896) ) ) {
                        result[0] += -0.09501663733723618;
                      } else {
                        result[0] += -0.017585682178162042;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.718933820724488193) ) ) {
                      result[0] += -0.009943110796808154;
                    } else {
                      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                        if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                          result[0] += -0.09974817357557114;
                        } else {
                          result[0] += 0.11514159316334596;
                        }
                      } else {
                        result[0] += -0.03719454424925831;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                      result[0] += -0.016914561822695986;
                    } else {
                      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += -0.09066461540784748;
                      } else {
                        result[0] += 0.0017480718860198683;
                      }
                    }
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.456172943115236151) ) ) {
                result[0] += -0.025196361082279198;
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.91355895996093928) ) ) {
                  result[0] += 0.001239284017710895;
                } else {
                  result[0] += -0.02059681077382256;
                }
              }
            } else {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.0941517276678745;
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.607751369476319248) ) ) {
                    result[0] += -0.006877529401987143;
                  } else {
                    result[0] += -0.046571164767234004;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.605039834976196733) ) ) {
                  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.011833465253018566;
                  } else {
                    result[0] += -0.061037797746238326;
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.357691764831543413) ) ) {
                    result[0] += -0.09226965871785076;
                  } else {
                    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.302512168884278232) ) ) {
                        result[0] += -0.035577847255453585;
                      } else {
                        result[0] += -0.006204191772382201;
                      }
                    } else {
                      result[0] += 0.0049152221296577875;
                    }
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.64270687103271662) ) ) {
            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.718933820724488193) ) ) {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.673553824424744096) ) ) {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)176.0000000000000284) ) ) {
                    result[0] += 0.05735927288695504;
                  } else {
                    result[0] += -0.1074728342810642;
                  }
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
                    if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.431880712509156162) ) ) {
                        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.651049375534058505) ) ) {
                          result[0] += -0.03754908222402453;
                        } else {
                          result[0] += 0.03654045890349662;
                        }
                      } else {
                        if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += -0.18815027789749073;
                        } else {
                          result[0] += 0.06296265692983687;
                        }
                      }
                    } else {
                      result[0] += 0.06515313769435875;
                    }
                  } else {
                    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += -0.07567702070082355;
                    } else {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.036049604415894443) ) ) {
                        if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += 0.0027026225461399414;
                        } else {
                          result[0] += -0.06524539312339614;
                        }
                      } else {
                        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                            result[0] += -0.11611835059402692;
                          } else {
                            result[0] += 0.047763283083056045;
                          }
                        } else {
                          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                            result[0] += -0.009769428345993411;
                          } else {
                            result[0] += 0.10098198452477164;
                          }
                        }
                      }
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += -0.04172445994337077;
                  } else {
                    result[0] += 0.040846283362967944;
                  }
                } else {
                  result[0] += 0.0771656354863883;
                }
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.827801465988160068) ) ) {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.03624371417476572;
                } else {
                  result[0] += 0.1138446031087122;
                }
              } else {
                result[0] += -0.04025805349035537;
              }
            }
          } else {
            result[0] += -0.11649025680408581;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
        result[0] += -0.0006896720185226842;
      } else {
        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)9.500000000000001776) ) ) {
          if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += -0.0702126278089653;
          } else {
            result[0] += -0.021092626162682995;
          }
        } else {
          result[0] += 0.0046339754889549115;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)0.8958797454833985485) ) ) {
    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)0.8958797454833985485) ) ) {
      result[0] += -0.08148450019013585;
    } else {
      result[0] += -0.0008539591987607796;
    }
  } else {
    if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
      result[0] += 0.000986873413039208;
    } else {
      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.909855604171753818) ) ) {
        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
          if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
            if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.255632162094117099) ) ) {
                if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.589026927947998269) ) ) {
                    result[0] += -0.018707819703043273;
                  } else {
                    result[0] += 0.02525901338819188;
                  }
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.141444921493531162) ) ) {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.342454433441162998) ) ) {
                      result[0] += 0.013362769222443166;
                    } else {
                      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                        result[0] += -0.007939694321539533;
                      } else {
                        result[0] += -0.0701639555008621;
                      }
                    }
                  } else {
                    result[0] += 0.0896379024435368;
                  }
                }
              } else {
                result[0] += -0.042232646064532905;
              }
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.302512168884278232) ) ) {
                if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                  result[0] += -0.1021230018544374;
                } else {
                  result[0] += 0.01889519049376779;
                }
              } else {
                if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                  result[0] += -0.04938661722899182;
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.909855604171753818) ) ) {
                    result[0] += 0.011644851287835494;
                  } else {
                    result[0] += -0.09912046178935482;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.651049375534058505) ) ) {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.154959201812744585) ) ) {
                    result[0] += 0.11148176118148057;
                  } else {
                    result[0] += 0.005352935938304033;
                  }
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.694163918495178667) ) ) {
                    result[0] += -0.021605088868812718;
                  } else {
                    if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.003849603253941238;
                    } else {
                      result[0] += 0.05052766790741883;
                    }
                  }
                }
              } else {
                result[0] += -0.035726084868403805;
              }
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.302512168884278232) ) ) {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.467917680740357333) ) ) {
                    result[0] += -0.07878121445924209;
                  } else {
                    result[0] += 0.06312960685195841;
                  }
                } else {
                  result[0] += 0.10596094145834277;
                }
              } else {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.540854334831238237) ) ) {
                  if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                    result[0] += 0.0684758221515867;
                  } else {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.917405366897583452) ) ) {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.249904870986938921) ) ) {
                        result[0] += 0.01651381870638644;
                      } else {
                        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.602003335952759233) ) ) {
                          result[0] += -0.11496997363978952;
                        } else {
                          result[0] += -0.016276348056045394;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.276966691017151323) ) ) {
                        result[0] += 0.13932464641760745;
                      } else {
                        result[0] += -0.03380778651414972;
                      }
                    }
                  }
                } else {
                  result[0] += -0.017850523876070754;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.484580039978028232) ) ) {
            if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)5.500000000000000888) ) ) {
              result[0] += -0.024981531461084224;
            } else {
              result[0] += -0.0708185990042526;
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)208.0000000000000284) ) ) {
                  result[0] += 0.07870573811752041;
                } else {
                  result[0] += -0.10415645182093963;
                }
              } else {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += 0.028509469087837597;
                  } else {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.48738741874694913) ) ) {
                      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
                        result[0] += -0.06924867713369096;
                      } else {
                        result[0] += -0.008937224870799398;
                      }
                    } else {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.924581527709961826) ) ) {
                        result[0] += 0.06413195240048838;
                      } else {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.947818994522095615) ) ) {
                          result[0] += -0.19030947140800578;
                        } else {
                          result[0] += 0.011426518298139928;
                        }
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)208.0000000000000284) ) ) {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += 0.026155993030493936;
                    } else {
                      result[0] += -0.06253373540423553;
                    }
                  } else {
                    result[0] += -0.07217035942675422;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.302512168884278232) ) ) {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)208.0000000000000284) ) ) {
                  if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2727.500000000000455) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.284418344497681552) ) ) {
                      result[0] += -0.07971882111794609;
                    } else {
                      if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                        if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                          result[0] += -0.06910523580152687;
                        } else {
                          result[0] += -0.009201374916517001;
                        }
                      } else {
                        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                          result[0] += -0.0027759123160811926;
                        } else {
                          result[0] += -0.2083324253862085;
                        }
                      }
                    }
                  } else {
                    result[0] += -0.10749648619131101;
                  }
                } else {
                  result[0] += 0.009727726789544644;
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.607751369476319248) ) ) {
                  if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                    result[0] += 0.033189702848552535;
                  } else {
                    result[0] += -0.00939817547278321;
                  }
                } else {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                    if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                      result[0] += -0.008630442612310186;
                    } else {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.44140100479126021) ) ) {
                        result[0] += -0.05570074121384283;
                      } else {
                        result[0] += 0.057735227813178684;
                      }
                    }
                  } else {
                    result[0] += -0.06443403521332519;
                  }
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.484580039978028232) ) ) {
            result[0] += 0.05027550790334165;
          } else {
            result[0] += -0.04061942871805668;
          }
        } else {
          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
            result[0] += -0.03333180620216176;
          } else {
            if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
              if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += 0.016557655500803663;
              } else {
                result[0] += -0.07294170887432669;
              }
            } else {
              result[0] += 0.028271524530033128;
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)0.8958797454833985485) ) ) {
    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)0.8958797454833985485) ) ) {
      result[0] += -0.08148450019013585;
    } else {
      result[0] += -0.0008539591987607796;
    }
  } else {
    if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)1.497866153717041238) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)15.26177501678466975) ) ) {
            result[0] += 0.008198591405137295;
          } else {
            if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
              result[0] += 0.17826938318686342;
            } else {
              result[0] += 0.024906609696418273;
            }
          }
        } else {
          if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
              if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.07048937197468902;
                } else {
                  result[0] += -0.022968878253940263;
                }
              } else {
                result[0] += 0.0011630684029893211;
              }
            } else {
              if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                  result[0] += -0.0015053548977484298;
                } else {
                  result[0] += -0.10336784834627742;
                }
              } else {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)1.354025125503540261) ) ) {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.232423543930054599) ) ) {
                      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
                        result[0] += 0.00324032667637378;
                      } else {
                        result[0] += -0.0731688101055281;
                      }
                    } else {
                      if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                        result[0] += -0.05265056763748145;
                      } else {
                        if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += 0.05232321490512803;
                        } else {
                          result[0] += 0.016481066547542098;
                        }
                      }
                    }
                  } else {
                    result[0] += 0.19436728423062505;
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.718933820724488193) ) ) {
                      result[0] += -0.01241362121571446;
                    } else {
                      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                        if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                          result[0] += -0.09941935707385369;
                        } else {
                          result[0] += 0.10963934860669705;
                        }
                      } else {
                        result[0] += -0.012420148269769723;
                      }
                    }
                  } else {
                    result[0] += -0.037822250812888;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.456172943115236151) ) ) {
                result[0] += -0.024783851914290504;
              } else {
                result[0] += 0.0005643647604809326;
              }
            } else {
              if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.09441567758959826;
                } else {
                  result[0] += -0.017264650548986554;
                }
              } else {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.605039834976196733) ) ) {
                    result[0] += 0.006219554731600017;
                  } else {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.302512168884278232) ) ) {
                      if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.284418344497681552) ) ) {
                        result[0] += -0.042059592222582806;
                      } else {
                        result[0] += 0.021882881173777943;
                      }
                    } else {
                      result[0] += -0.006238292774774203;
                    }
                  }
                } else {
                  result[0] += 0.016666043912730646;
                }
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.849175214767456943) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.029068946838379794) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.64270687103271662) ) ) {
                result[0] += 0.06286691482720222;
              } else {
                result[0] += -0.1291143264346711;
              }
            } else {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.276817798614503729) ) ) {
                  result[0] += 0.12054223859073042;
                } else {
                  if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.24005979159697916;
                  } else {
                    result[0] += -0.023548968891840064;
                  }
                }
              } else {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.07729864120483576) ) ) {
                    result[0] += 0.1000188371000817;
                  } else {
                    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += -0.08364644896259579;
                    } else {
                      result[0] += 0.008124503726829527;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += 0.0946516817565173;
                  } else {
                    result[0] += -0.12179891844230542;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
              result[0] += 0.09155107137558915;
            } else {
              result[0] += -0.07282827925062747;
            }
          }
        } else {
          if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += 0.007029727189655741;
            } else {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += -0.10939590234956387;
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.142747402191162998) ) ) {
                  result[0] += -0.026972595062428568;
                } else {
                  result[0] += 0.07623692537992047;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                  if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.208071470260621005) ) ) {
                    result[0] += 0.10205937090699926;
                  } else {
                    result[0] += -0.0527808438690406;
                  }
                } else {
                  result[0] += -0.1007619724820361;
                }
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.56941866874694913) ) ) {
                  result[0] += -0.053124670013543075;
                } else {
                  result[0] += -0.16347211756524607;
                }
              }
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.718933820724488193) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.484580039978028232) ) ) {
                  result[0] += -0.038242373602977865;
                } else {
                  if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += 0.0030631755930301654;
                  } else {
                    result[0] += 0.10817622048867188;
                  }
                }
              } else {
                result[0] += 0.11563663482868543;
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
        result[0] += -0.0008532417213048649;
      } else {
        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)9.500000000000001776) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.938988685607911933) ) ) {
            result[0] += 0.006445991672291993;
          } else {
            result[0] += -0.03155885528561498;
          }
        } else {
          if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
            result[0] += -0.015368986823523306;
          } else {
            if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)10.50000000000000178) ) ) {
              if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.302512168884278232) ) ) {
                  result[0] += -0.015903277496992687;
                } else {
                  result[0] += 0.08789494053394303;
                }
              } else {
                result[0] += 0.1251767771780334;
              }
            } else {
              result[0] += -0.025110942221846524;
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)0.8958797454833985485) ) ) {
    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)0.8958797454833985485) ) ) {
      result[0] += -0.08148450019012239;
    } else {
      result[0] += -0.0008539591987607796;
    }
  } else {
    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)15.26177501678466975) ) ) {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
        result[0] += -0.00024522457478260107;
      } else {
        if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
          if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.938867926597595659) ) ) {
              result[0] += -0.08856165208402057;
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.671854496002199042) ) ) {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += 0.0991448368489866;
                  } else {
                    result[0] += 0.018796158620363212;
                  }
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
                    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                      if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)176.0000000000000284) ) ) {
                        if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.835998296737671787) ) ) {
                          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.694163918495178667) ) ) {
                            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.172047138214112216) ) ) {
                              result[0] += 0.060378834082425115;
                            } else {
                              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.255632162094117099) ) ) {
                                result[0] += -0.10079206868878213;
                              } else {
                                result[0] += 0.0390807456625967;
                              }
                            }
                          } else {
                            result[0] += 0.06306583652593023;
                          }
                        } else {
                          result[0] += -0.061746454562343504;
                        }
                      } else {
                        result[0] += -0.09415850477941401;
                      }
                    } else {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.431880712509156162) ) ) {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.142747402191162998) ) ) {
                          result[0] += -0.03313538578351461;
                        } else {
                          result[0] += 0.09963662327143072;
                        }
                      } else {
                        result[0] += -0.19565165023419237;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.302512168884278232) ) ) {
                        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                          if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                            result[0] += -0.11905430483581754;
                          } else {
                            result[0] += 0.14218643939171993;
                          }
                        } else {
                          result[0] += 0.13428312998035438;
                        }
                      } else {
                        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.81807899475097834) ) ) {
                          if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
                            result[0] += 0.03503398245399759;
                          } else {
                            result[0] += -0.07131775638475245;
                          }
                        } else {
                          if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                            result[0] += -0.045727452665878084;
                          } else {
                            result[0] += -0.16369003551459;
                          }
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.036049604415894443) ) ) {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.397998809814454013) ) ) {
                          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)176.0000000000000284) ) ) {
                            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                              if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
                                result[0] += -0.054059080616271044;
                              } else {
                                result[0] += -0.29230000683126833;
                              }
                            } else {
                              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.56941866874694913) ) ) {
                                result[0] += -0.06883903721406505;
                              } else {
                                result[0] += 0.012681791453770287;
                              }
                            }
                          } else {
                            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.329314231872559482) ) ) {
                                result[0] += 0.072482382645762;
                              } else {
                                if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                                  result[0] += 0.10810405577692313;
                                } else {
                                  result[0] += 0.45734363903816855;
                                }
                              }
                            } else {
                              result[0] += -0.07310650692016678;
                            }
                          }
                        } else {
                          if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)176.0000000000000284) ) ) {
                                result[0] += 0.029743084490216423;
                              } else {
                                result[0] += -0.0793119136481169;
                              }
                            } else {
                              result[0] += -0.05667028875843292;
                            }
                          } else {
                            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                              result[0] += -0.020354936552262545;
                            } else {
                              result[0] += 0.05527718197458059;
                            }
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.363266706466675693) ) ) {
                              result[0] += 0.03463145734766398;
                            } else {
                              result[0] += -0.11566843747419385;
                            }
                          } else {
                            result[0] += 0.10654129229145362;
                          }
                        } else {
                          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                            if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                              result[0] += 0.03067319939818617;
                            } else {
                              result[0] += -0.1569695545040951;
                            }
                          } else {
                            result[0] += 0.0861295243512854;
                          }
                        }
                      }
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.993164777755738193) ) ) {
                    if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)176.0000000000000284) ) ) {
                      result[0] += 0.08385740628709396;
                    } else {
                      result[0] += -0.08099429737801474;
                    }
                  } else {
                    result[0] += -0.005253291877607804;
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.909855604171753818) ) ) {
                    result[0] += -0.012295594552193654;
                  } else {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                      result[0] += 0.1404145134524482;
                    } else {
                      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                          result[0] += 0.1238169773977538;
                        } else {
                          result[0] += -0.08749588686475143;
                        }
                      } else {
                        result[0] += 0.11242217026653707;
                      }
                    }
                  }
                }
              }
            }
          } else {
            result[0] += -0.10994672189384835;
          }
        } else {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
            result[0] += -0.06327092541670824;
          } else {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.607751369476319248) ) ) {
                result[0] += -0.025837638761223025;
              } else {
                result[0] += 0.051233084992308366;
              }
            } else {
              result[0] += -0.08311491699129013;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.699081301689148393) ) ) {
          result[0] += 0.19505284407467544;
        } else {
          result[0] += 0.0814343892742892;
        }
      } else {
        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += -0.05564256586803806;
          } else {
            if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += 0.034370585881550515;
              } else {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += 0.05495516414064277;
                } else {
                  result[0] += -0.021583605239229517;
                }
              }
            } else {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                result[0] += 0.10604200284401302;
              } else {
                result[0] += -0.006628038161336856;
              }
            }
          }
        } else {
          result[0] += -0.05153521457432736;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)0.8958797454833985485) ) ) {
    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)0.8958797454833985485) ) ) {
      result[0] += -0.08148450019013585;
    } else {
      result[0] += -0.0008539591987607796;
    }
  } else {
    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)15.26177501678466975) ) ) {
        result[0] += -0.00028983779038836173;
      } else {
        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.699081301689148393) ) ) {
            result[0] += 0.19733262270260152;
          } else {
            result[0] += 0.08598895938776441;
          }
        } else {
          result[0] += -0.006930276336788471;
        }
      }
    } else {
      if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
        if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
            if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.64270687103271662) ) ) {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.802901029586792436) ) ) {
                  if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.013227727253147449;
                  } else {
                    result[0] += 0.10046560819024614;
                  }
                } else {
                  if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
                      if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.780892848968506748) ) ) {
                          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.715336322784424716) ) ) {
                            result[0] += -0.03266143563312159;
                          } else {
                            result[0] += -0.1893534168737463;
                          }
                        } else {
                          result[0] += 0.019553395306238418;
                        }
                      } else {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
                          result[0] += -0.03893134425614404;
                        } else {
                          result[0] += 0.11144893458191985;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.051912069320679599) ) ) {
                          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.284418344497681552) ) ) {
                            result[0] += -0.14172655754908745;
                          } else {
                            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
                              result[0] += -0.03448277600761162;
                            } else {
                              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)176.0000000000000284) ) ) {
                                result[0] += 0.13340058540401684;
                              } else {
                                result[0] += 0.0017658864882695536;
                              }
                            }
                          }
                        } else {
                          result[0] += 0.11054692652504183;
                        }
                      } else {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.255632162094117099) ) ) {
                          result[0] += 0.06586753490190819;
                        } else {
                          result[0] += -0.020634100140061576;
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.827801465988160068) ) ) {
                      result[0] += 0.09845482857892926;
                    } else {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.607751369476319248) ) ) {
                        result[0] += -0.13153404920531134;
                      } else {
                        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.067782521247864214) ) ) {
                          result[0] += -0.14289564187213433;
                        } else {
                          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.511434078216553178) ) ) {
                            result[0] += 0.11240328689794657;
                          } else {
                            if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                                result[0] += -0.14469482996092625;
                              } else {
                                result[0] += -0.00015242715889849333;
                              }
                            } else {
                              result[0] += 0.046065034966791235;
                            }
                          }
                        }
                      }
                    }
                  }
                }
              } else {
                result[0] += -0.1228394206218566;
              }
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.700598716735840066) ) ) {
                result[0] += 0.07554725535805071;
              } else {
                if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.06826721726146132;
                } else {
                  result[0] += 0.001596313784424726;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.051912069320679599) ) ) {
                if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.05726614796795894;
                } else {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.827801465988160068) ) ) {
                      result[0] += -0.04520311653210603;
                    } else {
                      result[0] += 0.06629297405310838;
                    }
                  } else {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.467917680740357333) ) ) {
                      result[0] += -0.09039781331179415;
                    } else {
                      result[0] += 0.026469258924401298;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.373361587524414951) ) ) {
                      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                        result[0] += 0.011317859910463975;
                      } else {
                        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                          result[0] += 0.05511725113103615;
                        } else {
                          if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.172047138214112216) ) ) {
                              result[0] += 0.21048286229214683;
                            } else {
                              result[0] += 0.624010313870113;
                            }
                          } else {
                            result[0] += -0.07348989920714208;
                          }
                        }
                      }
                    } else {
                      result[0] += -0.023206032968610974;
                    }
                  } else {
                    result[0] += -0.07662642457943437;
                  }
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.43742394447326749) ) ) {
                    result[0] += 0.017905964150671264;
                  } else {
                    result[0] += 0.10314065428595612;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.208071470260621005) ) ) {
                      result[0] += 0.08081190642900508;
                    } else {
                      result[0] += -0.05268444628476091;
                    }
                  } else {
                    result[0] += -0.09971482072628235;
                  }
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.56941866874694913) ) ) {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += -0.1752732720410106;
                      } else {
                        if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                          result[0] += 0.042673040196361686;
                        } else {
                          result[0] += -0.09838593239080562;
                        }
                      }
                    } else {
                      result[0] += -0.15851243388148656;
                    }
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.516936540603638583) ) ) {
                      if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += -0.12789971847276013;
                      } else {
                        result[0] += -0.38409635855143487;
                      }
                    } else {
                      result[0] += -0.06595880015524593;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.718933820724488193) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.484580039978028232) ) ) {
                    result[0] += -0.03610503846164619;
                  } else {
                    if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                      result[0] += 0.00448660362677208;
                    } else {
                      result[0] += 0.09327354187423834;
                    }
                  }
                } else {
                  result[0] += 0.10004955506414569;
                }
              }
            }
          }
        } else {
          result[0] += -0.10596089997732126;
        }
      } else {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
          result[0] += -0.06093279191486495;
        } else {
          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
            result[0] += 0.020957042153928254;
          } else {
            result[0] += -0.08156996526807514;
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)0.8958797454833985485) ) ) {
    if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.589026927947998269) ) ) {
      result[0] += -0.008902491364181775;
    } else {
      if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)12.00000000000000178) ) ) {
        result[0] += -0.08643572833185864;
      } else {
        result[0] += -0.0073434639533375345;
      }
    }
  } else {
    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
      if ( UNLIKELY(  (data[43].missing != -1) && (data[43].fvalue <= (double)-1.00000001800250948e-35) ) ) {
        result[0] += 0.0008408412245219911;
      } else {
        if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
          result[0] += -0.0036988004197784415;
        } else {
          if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
            if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.361115694046021396) ) ) {
                result[0] += 0.08896894246573779;
              } else {
                result[0] += -0.0631424051721957;
              }
            } else {
              if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.02604460716247603) ) ) {
                      result[0] += 0.11805836910448315;
                    } else {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.94957673549652144) ) ) {
                        result[0] += -0.19115879539651648;
                      } else {
                        result[0] += -0.04278932343595934;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.827801465988160068) ) ) {
                      result[0] += 0.18762385585059888;
                    } else {
                      result[0] += -0.009217826300468036;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.516936540603638583) ) ) {
                      result[0] += 0.06666082396556168;
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.154959201812744585) ) ) {
                        result[0] += -0.15519071442706953;
                      } else {
                        result[0] += 0.009681274082467365;
                      }
                    }
                  } else {
                    result[0] += -0.18182979072526093;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += 0.06959091609597412;
                } else {
                  result[0] += 0.017331517414348876;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.605039834976196733) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.827801465988160068) ) ) {
                result[0] += 0.00014963399177996784;
              } else {
                result[0] += 0.03711663102186324;
              }
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.795762062072754794) ) ) {
                if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
                  result[0] += -0.021579702512534788;
                } else {
                  result[0] += -0.08771244574012178;
                }
              } else {
                if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.667095184326172763) ) ) {
                    result[0] += -0.016132270112762824;
                  } else {
                    result[0] += -0.28106979335646315;
                  }
                } else {
                  result[0] += 0.01341072874848402;
                }
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.64270687103271662) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)176.0000000000000284) ) ) {
                result[0] += 0.05366935079157531;
              } else {
                result[0] += -0.09391887949965039;
              }
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.431880712509156162) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.142747402191162998) ) ) {
                  result[0] += -0.01734269598914299;
                } else {
                  result[0] += 0.09733839639405634;
                }
              } else {
                result[0] += -0.14535321641146837;
              }
            }
          } else {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.108761310577394354) ) ) {
                result[0] += 0.11569661042332746;
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.397998809814454013) ) ) {
                  result[0] += 0.04537741265602602;
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.607751369476319248) ) ) {
                    result[0] += -0.07598166379047704;
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.255632162094117099) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.07729864120483576) ) ) {
                        result[0] += 0.14855586004840093;
                      } else {
                        result[0] += -0.011243364993008344;
                      }
                    } else {
                      result[0] += -0.11241344546692118;
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.90474271774292081) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.397998809814454013) ) ) {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)176.0000000000000284) ) ) {
                    result[0] += -0.06608005793730788;
                  } else {
                    if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                        result[0] += 0.37814440499836754;
                      } else {
                        result[0] += 0.0845395048919414;
                      }
                    } else {
                      result[0] += -0.07238968717372733;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                        if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                          if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.870983839035034624) ) ) {
                            result[0] += 0.01435734703723505;
                          } else {
                            result[0] += -0.09196532370056534;
                          }
                        } else {
                          result[0] += 0.043009458804035366;
                        }
                      } else {
                        result[0] += -0.0463098948087388;
                      }
                    } else {
                      result[0] += -0.09933628738315002;
                    }
                  } else {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.484580039978028232) ) ) {
                      result[0] += -0.07779892871554288;
                    } else {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.607751369476319248) ) ) {
                        if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)176.0000000000000284) ) ) {
                              result[0] += 0.14592100920884685;
                            } else {
                              result[0] += -0.013647085532226006;
                            }
                          } else {
                            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                              result[0] += 0.0487592105290863;
                            } else {
                              result[0] += -0.0674718910635162;
                            }
                          }
                        } else {
                          result[0] += -0.06504050351666192;
                        }
                      } else {
                        result[0] += 0.06475977702181;
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += -0.025877874264739916;
                  } else {
                    result[0] += 0.09214669615488003;
                  }
                } else {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += -0.01282176788113417;
                    } else {
                      if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                          result[0] += -0.09416474600366807;
                        } else {
                          result[0] += 0.06793928922479242;
                        }
                      } else {
                        result[0] += 0.09406271011041761;
                      }
                    }
                  } else {
                    result[0] += -0.10548778043119628;
                  }
                }
              }
            }
          }
        } else {
          result[0] += -0.10801548119961456;
        }
      } else {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
          result[0] += -0.05988084726808819;
        } else {
          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
            result[0] += 0.017969109749953586;
          } else {
            result[0] += -0.07995146086650062;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)0.8958797454833985485) ) ) {
      if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.589026927947998269) ) ) {
        result[0] += -0.008902491364181109;
      } else {
        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
          result[0] += -0.026350175147518935;
        } else {
          result[0] += -0.09980251447386485;
        }
      }
    } else {
      if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.242453336715698464) ) ) {
          if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)257681260544.0000305) ) ) {
            if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.002088336080024927;
            } else {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.827801465988160068) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.7512402534484881) ) ) {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                    result[0] += -0.0018184470507147381;
                  } else {
                    if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.019405965649845477;
                    } else {
                      result[0] += 0.003965726292105959;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                    result[0] += 0.006573539842926229;
                  } else {
                    result[0] += -0.013156888279305917;
                  }
                }
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.206118345260621005) ) ) {
                  result[0] += 0.036864733421582856;
                } else {
                  result[0] += 0.10899473438187546;
                }
              }
            }
          } else {
            result[0] += 0.029738620741645162;
          }
        } else {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.445705175399781162) ) ) {
            result[0] += 0.08895756615760443;
          } else {
            result[0] += -0.008581743149625527;
          }
        }
      } else {
        if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.947818994522095615) ) ) {
            result[0] += -0.002078525574614811;
          } else {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
              result[0] += -0.040999199901874894;
            } else {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.172047138214112216) ) ) {
                  result[0] += 0.0247673029926893;
                } else {
                  result[0] += -0.02304083928630476;
                }
              } else {
                if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)2.087193608283997026) ) ) {
                  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)4.500000000000000888) ) ) {
                    result[0] += 0.022632391930889544;
                  } else {
                    result[0] += -0.07803471809188492;
                  }
                } else {
                  result[0] += 0.10311628240126647;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.617236852645874912) ) ) {
              if ( LIKELY( !(data[10].missing != -1) || (data[10].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.07729864120483576) ) ) {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
                    result[0] += 0.013010919959091448;
                  } else {
                    result[0] += 0.08890286122265187;
                  }
                } else {
                  result[0] += -0.003616381400176269;
                }
              } else {
                result[0] += -0.08126805691929427;
              }
            } else {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += -0.02270180403129664;
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.644374847412110263) ) ) {
                  if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                    result[0] += 0.09594715346140598;
                  } else {
                    result[0] += 0.02497949288268371;
                  }
                } else {
                  if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                    result[0] += 0.06394328440688334;
                  } else {
                    result[0] += -0.09644428851368053;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.158952236175537998) ) ) {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                result[0] += 0.010672989889805454;
              } else {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.909855604171753818) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.172047138214112216) ) ) {
                      result[0] += -0.06347633856784012;
                    } else {
                      result[0] += 0.00877187037086456;
                    }
                  } else {
                    if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2415.000000000000455) ) ) {
                      result[0] += 0.041760039375687494;
                    } else {
                      result[0] += -0.1129894241148681;
                    }
                  }
                } else {
                  result[0] += -0.17068953346006274;
                }
              }
            } else {
              result[0] += 0.0002539162854065382;
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.142747402191162998) ) ) {
        if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.938867926597595659) ) ) {
          result[0] += -0.08958215263357583;
        } else {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.64270687103271662) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
              if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.849175214767456943) ) ) {
                  result[0] += -0.02131553588852903;
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.208071470260621005) ) ) {
                    result[0] += -0.018095170770137566;
                  } else {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                      result[0] += 0.11839016354068728;
                    } else {
                      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                        result[0] += -0.06663408187001514;
                      } else {
                        result[0] += 0.05982436917915911;
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
                  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += -0.00794308123133769;
                  } else {
                    result[0] += 0.07842434836790015;
                  }
                } else {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
                    if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                      result[0] += 0.018702300457426765;
                    } else {
                      result[0] += -0.07978115015515613;
                    }
                  } else {
                    result[0] += -0.12189302587149958;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.19876670837402521) ) ) {
                result[0] += 0.060362549317270156;
              } else {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
                  result[0] += 0.021501664495370398;
                } else {
                  result[0] += -0.09127725920030108;
                }
              }
            }
          } else {
            result[0] += -0.10489249430923855;
          }
        }
      } else {
        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
            result[0] += -0.019596324448256253;
          } else {
            result[0] += 0.10954304882718623;
          }
        } else {
          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
            result[0] += -0.033112650026412324;
          } else {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += 0.03827507776737222;
              } else {
                result[0] += -0.13507218310606653;
              }
            } else {
              result[0] += 0.07931425650703068;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += -0.025167927645922645;
          } else {
            result[0] += 0.12092803449751605;
          }
        } else {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
            result[0] += -0.05566011119234881;
          } else {
            result[0] += 0.042810796113094296;
          }
        }
      } else {
        result[0] += -0.08086987414692026;
      }
    }
  }
  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)0.8958797454833985485) ) ) {
    if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.589026927947998269) ) ) {
      result[0] += -0.008902491364181123;
    } else {
      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
        result[0] += -0.026350175147518935;
      } else {
        result[0] += -0.09980251447386485;
      }
    }
  } else {
    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
      if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.242453336715698464) ) ) {
          if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)257681260544.0000305) ) ) {
            if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.0020144912935427086;
            } else {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.827801465988160068) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.7512402534484881) ) ) {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                    result[0] += -0.0018599916576838623;
                  } else {
                    if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.019334651739264345;
                    } else {
                      result[0] += 0.003895280731934965;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                    result[0] += 0.006440759718881835;
                  } else {
                    result[0] += -0.012570018147184715;
                  }
                }
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.206118345260621005) ) ) {
                  result[0] += 0.03511161640792249;
                } else {
                  result[0] += 0.11103407266829601;
                }
              }
            }
          } else {
            result[0] += 0.02891184068069566;
          }
        } else {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.445705175399781162) ) ) {
            result[0] += 0.08659027885727995;
          } else {
            result[0] += -0.0046853892769759065;
          }
        }
      } else {
        if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
            result[0] += -0.03950431440517284;
          } else {
            if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)8816427008.000001907) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.65098142623901545) ) ) {
                result[0] += 0.07103916066034016;
              } else {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += -0.015294976506678644;
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.607751369476319248) ) ) {
                    result[0] += -0.017727963217612924;
                  } else {
                    if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += 0.10976608697295126;
                    } else {
                      result[0] += 0.024633491966802926;
                    }
                  }
                }
              }
            } else {
              result[0] += -0.003550160096549355;
            }
          }
        } else {
          if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)9.500000000000001776) ) ) {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.909855604171753818) ) ) {
                result[0] += 0.004350438429673478;
              } else {
                result[0] += -0.021453050837345618;
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.284418344497681552) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.215408444404602495) ) ) {
                  result[0] += -0.004861488253291293;
                } else {
                  result[0] += -0.03354085845835699;
                }
              } else {
                result[0] += 0.002472429759148494;
              }
            }
          } else {
            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += 0.030825195387976845;
            } else {
              result[0] += -0.029030412876440065;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.142747402191162998) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
            if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.651049375534058505) ) ) {
                result[0] += -0.030141023969074972;
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.063810348510743076) ) ) {
                  if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.088880300521851474) ) ) {
                      result[0] += -0.07996205190513127;
                    } else {
                      result[0] += 0.05196240205843168;
                    }
                  } else {
                    result[0] += 0.09645143339561592;
                  }
                } else {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.097527027130127841) ) ) {
                      result[0] += -0.14908721741599656;
                    } else {
                      result[0] += 0.07786710767228196;
                    }
                  } else {
                    result[0] += -0.2101588570818278;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.835998296737671787) ) ) {
                result[0] += 0.05646616989371464;
              } else {
                result[0] += -0.057894620374931344;
              }
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.607751369476319248) ) ) {
              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.431901693344116655) ) ) {
                result[0] += 0.11123215870371324;
              } else {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                  result[0] += -0.029957526681214555;
                } else {
                  if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.172047138214112216) ) ) {
                      result[0] += 0.23122167607285748;
                    } else {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.158952236175537998) ) ) {
                        result[0] += -0.08348075259382773;
                      } else {
                        result[0] += 0.13708442129859807;
                      }
                    }
                  } else {
                    result[0] += -0.05791593518344132;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.719506263732911933) ) ) {
                  result[0] += 0.03903282366687825;
                } else {
                  result[0] += -0.055382564721416776;
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.87254357337951749) ) ) {
                  if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += -0.17242024999983221;
                  } else {
                    result[0] += -0.004234699210567786;
                  }
                } else {
                  if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.909855604171753818) ) ) {
                            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                              result[0] += 0.07735540465709617;
                            } else {
                              result[0] += -0.11396715108554303;
                            }
                          } else {
                            result[0] += 0.10928205339504403;
                          }
                        } else {
                          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                            result[0] += -0.0944144391183858;
                          } else {
                            result[0] += 0.015171730416835078;
                          }
                        }
                      } else {
                        result[0] += 0.06820596321240659;
                      }
                    } else {
                      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                        result[0] += -0.06913904474296657;
                      } else {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.088880300521851474) ) ) {
                          result[0] += -0.01243423517725214;
                        } else {
                          result[0] += 0.14410796123227723;
                        }
                      }
                    }
                  } else {
                    result[0] += -0.09797344569192125;
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.431880712509156162) ) ) {
            result[0] += 0.0639981560736991;
          } else {
            result[0] += 0.018004669854389903;
          }
        }
      } else {
        if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
            result[0] += 0.050667081508062886;
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
              result[0] += -0.053271645546448566;
            } else {
              result[0] += 0.0368228576035473;
            }
          }
        } else {
          result[0] += -0.07924493903121056;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)0.8958797454833985485) ) ) {
    if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.589026927947998269) ) ) {
      result[0] += -0.008902491364181123;
    } else {
      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
        result[0] += -0.026350175147518935;
      } else {
        result[0] += -0.09980251447386485;
      }
    }
  } else {
    if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
      result[0] += 0.001991760246333054;
    } else {
      if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.088880300521851474) ) ) {
          if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)8816427008.000001907) ) ) {
            if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.95478391647339045) ) ) {
                result[0] += -0.09463538662080939;
              } else {
                result[0] += 0.05642524683600504;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.87254357337951749) ) ) {
                result[0] += 0.13927106926368257;
              } else {
                if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.07836049750082313;
                } else {
                  result[0] += 0.03194151997731666;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.21969318389892756) ) ) {
                  if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.127004146575928623) ) ) {
                    result[0] += -0.002420769961102325;
                  } else {
                    if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.05635608875443568;
                    } else {
                      result[0] += 0.005432916894930476;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                    if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.313157320022583896) ) ) {
                        if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)0.8958797454833985485) ) ) {
                          result[0] += -0.04460250891016962;
                        } else {
                          result[0] += -0.1784490961846207;
                        }
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
                          result[0] += -0.0044587827748504205;
                        } else {
                          if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                            result[0] += -0.02967380475998916;
                          } else {
                            result[0] += -0.07750582484604963;
                          }
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                        result[0] += 0.1836459858187288;
                      } else {
                        result[0] += -0.005828588257849525;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                      if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                        result[0] += -0.010863518195561741;
                      } else {
                        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.467917680740357333) ) ) {
                          if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)0.8958797454833985485) ) ) {
                            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.484580039978028232) ) ) {
                              result[0] += -0.042374050140848996;
                            } else {
                              result[0] += 0.08645022985897355;
                            }
                          } else {
                            result[0] += -0.09008189925786952;
                          }
                        } else {
                          result[0] += 0.10964128249650168;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.484580039978028232) ) ) {
                        result[0] += 0.09393622744651509;
                      } else {
                        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.778982400894165927) ) ) {
                          result[0] += -0.11515258021796249;
                        } else {
                          result[0] += 0.03260646205200639;
                        }
                      }
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.589026927947998269) ) ) {
                  result[0] += -0.05508175114153661;
                } else {
                  result[0] += 0.0635804967162;
                }
              }
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += -0.01755770035623586;
              } else {
                if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)6.000000000000000888) ) ) {
                    if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)7.500000000000000888) ) ) {
                      result[0] += 0.011141932866123042;
                    } else {
                      result[0] += -0.02624886246903292;
                    }
                  } else {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                      if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += 0.01156334854653663;
                      } else {
                        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.516936540603638583) ) ) {
                            result[0] += -0.050917968084127255;
                          } else {
                            result[0] += -0.16710114300790568;
                          }
                        } else {
                          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                            result[0] += 0.03251746017791028;
                          } else {
                            result[0] += -0.11916796813330391;
                          }
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.48918962478637873) ) ) {
                        result[0] += -0.008877188027477447;
                      } else {
                        result[0] += -0.06646992036655452;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.605039834976196733) ) ) {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.357691764831543413) ) ) {
                        result[0] += 0.11693531674198017;
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.154959201812744585) ) ) {
                          result[0] += 0.06075882988473862;
                        } else {
                          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.45958471298217951) ) ) {
                            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.142630577087403232) ) ) {
                              result[0] += 0.04146000969411157;
                            } else {
                              result[0] += -0.06797784877107486;
                            }
                          } else {
                            result[0] += -0.0961923739506961;
                          }
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.198464870452881303) ) ) {
                          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                            result[0] += -0.009844430674249206;
                          } else {
                            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.64270687103271662) ) ) {
                                result[0] += -0.10519275642393222;
                              } else {
                                result[0] += 0.13057951297425324;
                              }
                            } else {
                              result[0] += 0.1409930320530305;
                            }
                          }
                        } else {
                          result[0] += -0.02232554771997444;
                        }
                      } else {
                        if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += 0.0200491024267208;
                        } else {
                          result[0] += -0.06351724868542566;
                        }
                      }
                    }
                  } else {
                    result[0] += -0.001203072578372569;
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.108761310577394354) ) ) {
              result[0] += 0.08104276738974922;
            } else {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)40.00000000000000711) ) ) {
                result[0] += -0.0632072448501393;
              } else {
                if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.047341657346442405;
                } else {
                  if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)2.087193608283997026) ) ) {
                    result[0] += -0.00914470910456056;
                  } else {
                    result[0] += -0.058136758780409276;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
              if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)6.000000000000000888) ) ) {
                if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                  result[0] += 0.010852476153755614;
                } else {
                  result[0] += 0.10384208931746877;
                }
              } else {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                  result[0] += -0.0115182311561038;
                } else {
                  result[0] += -0.14723209060513573;
                }
              }
            } else {
              result[0] += 0.05983786303565613;
            }
          }
        }
      } else {
        result[0] += 0.0021480704124801487;
      }
    }
  }
  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)0.8958797454833985485) ) ) {
    if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.589026927947998269) ) ) {
      result[0] += -0.008902491364181775;
    } else {
      if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)137422176256.0000153) ) ) {
        result[0] += -0.026350175147518935;
      } else {
        result[0] += -0.09980251447386485;
      }
    }
  } else {
    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.242453336715698464) ) ) {
          if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)257681260544.0000305) ) ) {
            if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.0020350312355597445;
            } else {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.827801465988160068) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.7512402534484881) ) ) {
                  if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2415.000000000000455) ) ) {
                    if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.01910280876720942;
                    } else {
                      result[0] += 0.0037683159597560693;
                    }
                  } else {
                    result[0] += -0.0016453890583941835;
                  }
                } else {
                  result[0] += -0.006941555209923141;
                }
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.206118345260621005) ) ) {
                  result[0] += 0.034185531212593864;
                } else {
                  result[0] += 0.11351751387695536;
                }
              }
            }
          } else {
            result[0] += 0.028170385883442207;
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.41211462020874201) ) ) {
            result[0] += 0.08549361500732666;
          } else {
            result[0] += 0.001889471280955599;
          }
        }
      } else {
        if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.718933820724488193) ) ) {
            if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)8816427008.000001907) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.87254357337951749) ) ) {
                result[0] += 0.14242574929959628;
              } else {
                result[0] += 0.009120881022594033;
              }
            } else {
              result[0] += -0.0015574197266603656;
            }
          } else {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.467917680740357333) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.172047138214112216) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.534971714019776279) ) ) {
                    result[0] += 0.000470297833120022;
                  } else {
                    result[0] += -0.044167004929213674;
                  }
                } else {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                    result[0] += -0.0403679997014946;
                  } else {
                    result[0] += 0.03651096821595415;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.543407917022706854) ) ) {
                  result[0] += -0.002141574755971836;
                } else {
                  result[0] += -0.04427745388946943;
                }
              }
            } else {
              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.552972793579102007) ) ) {
                result[0] += -0.10318903837343588;
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
                  result[0] += -0.0592411689176589;
                } else {
                  if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.329314231872559482) ) ) {
                        result[0] += -0.01314299091780519;
                      } else {
                        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                          result[0] += 0.04057022957788919;
                        } else {
                          result[0] += -0.04331925677540113;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.516936540603638583) ) ) {
                          result[0] += -0.043447536641954845;
                        } else {
                          result[0] += -0.005189804660933644;
                        }
                      } else {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.051912069320679599) ) ) {
                          result[0] += -0.06490004074439594;
                        } else {
                          result[0] += 0.028820309811404272;
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)7.500000000000000888) ) ) {
                      result[0] += 0.01936356759406421;
                    } else {
                      result[0] += 0.15716464801295082;
                    }
                  }
                }
              }
            }
          }
        } else {
          result[0] += 0.0016278009927096285;
        }
      }
    } else {
      if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.88435244560241788) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
            if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)257681260544.0000305) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.651049375534058505) ) ) {
                result[0] += -0.03049201410680464;
              } else {
                if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.088880300521851474) ) ) {
                    result[0] += -0.05176643594838223;
                  } else {
                    result[0] += 0.039709474291860274;
                  }
                } else {
                  result[0] += 0.08892149446627326;
                }
              }
            } else {
              if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.835998296737671787) ) ) {
                result[0] += 0.05388037434239196;
              } else {
                result[0] += -0.05090323506922423;
              }
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.607751369476319248) ) ) {
              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.431901693344116655) ) ) {
                result[0] += 0.10134129623544115;
              } else {
                result[0] += -0.027270365419300294;
              }
            } else {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.719506263732911933) ) ) {
                  result[0] += 0.03826611937068966;
                } else {
                  result[0] += -0.05222017663540222;
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.87254357337951749) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.516936540603638583) ) ) {
                    result[0] += 0.038221446109283436;
                  } else {
                    result[0] += -0.15026027795468166;
                  }
                } else {
                  result[0] += 0.02933671513722981;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
            result[0] += -0.051692746715273824;
          } else {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += -0.00994174756218083;
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.431880712509156162) ) ) {
                  result[0] += 0.10880500510011615;
                } else {
                  result[0] += 0.04414478181848572;
                }
              }
            } else {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += 0.02978226273024532;
              } else {
                result[0] += -0.1103919179166138;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
          if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)137422176256.0000153) ) ) {
            result[0] += -0.08529731835135952;
          } else {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.445957899093628818) ) ) {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.088880300521851474) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.172047138214112216) ) ) {
                  result[0] += 0.09734534076528455;
                } else {
                  result[0] += -0.08166058264870969;
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.029068946838379794) ) ) {
                  result[0] += -0.024564057127098783;
                } else {
                  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)7.500000000000000888) ) ) {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.000308036804200107) ) ) {
                      result[0] += 0.18722966143640907;
                    } else {
                      result[0] += 0.5334489481700904;
                    }
                  } else {
                    result[0] += 0.034344575446362796;
                  }
                }
              }
            } else {
              result[0] += -0.050051549491067326;
            }
          }
        } else {
          if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)2.087193608283997026) ) ) {
            result[0] += -0.061891470453954794;
          } else {
            result[0] += 0.10278368725476882;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
    if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
      if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
        if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
            if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
                if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2565.000000000000455) ) ) {
                  result[0] += 0.005784965646440996;
                } else {
                  result[0] += -0.018550947393142572;
                }
              } else {
                result[0] += 0.037250567676261497;
              }
            } else {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)6.685099840164185458) ) ) {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)6.117118597030640537) ) ) {
                  result[0] += -0.0018925650603907043;
                } else {
                  result[0] += 0.019677739919037606;
                }
              } else {
                result[0] += -0.030258247889048098;
              }
            }
          } else {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                  result[0] += 0.001284284226525368;
                } else {
                  result[0] += -0.10425003333663607;
                }
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.232423543930054599) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.736135363578796831) ) ) {
                    result[0] += 0.03765842084168956;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.95906782150268732) ) ) {
                      result[0] += -0.05338355676896844;
                    } else {
                      if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)2.238668441772461382) ) ) {
                        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                            result[0] += -0.0068707373070026195;
                          } else {
                            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                              result[0] += 0.0489761581615506;
                            } else {
                              result[0] += -0.0007702285628314212;
                            }
                          }
                        } else {
                          result[0] += -0.061771272967890645;
                        }
                      } else {
                        result[0] += -0.11281437192323399;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                    if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.0932687585193051;
                    } else {
                      result[0] += -0.00991186111328902;
                    }
                  } else {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                        result[0] += -0.052726828424957184;
                      } else {
                        result[0] += 0.028212648709261184;
                      }
                    } else {
                      if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.276817798614503729) ) ) {
                          result[0] += -0.07345850372540984;
                        } else {
                          result[0] += 0.03182626923401371;
                        }
                      } else {
                        result[0] += -0.05661498363570288;
                      }
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.08831794335183381;
                } else {
                  result[0] += -0.07913136853517849;
                }
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.43742394447326749) ) ) {
                  result[0] += -0.021256864735285194;
                } else {
                  result[0] += -0.13177315268135292;
                }
              }
            }
          }
        } else {
          result[0] += -0.0033607194200602746;
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.433569431304932529) ) ) {
          if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.3407330513000506) ) ) {
              result[0] += -0.01721575310835083;
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += 0.04575949329413734;
                } else {
                  result[0] += -0.037437756340059955;
                }
              } else {
                result[0] += -0.0211306218965663;
              }
            }
          } else {
            if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.909855604171753818) ) ) {
                result[0] += 0.06476640659326487;
              } else {
                result[0] += -0.03703483053545251;
              }
            } else {
              result[0] += -0.0010744224260024256;
            }
          }
        } else {
          if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
            result[0] += -0.006663036485380589;
          } else {
            result[0] += 0.009964364945982584;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
        if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)1.242453336715698464) ) ) {
            result[0] += -0.07853247477078547;
          } else {
            if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.032233102421919284;
              } else {
                result[0] += -0.00487465507037488;
              }
            } else {
              if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)6.000000000000000888) ) ) {
                if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.605039834976196733) ) ) {
                    result[0] += 0.027043948799860686;
                  } else {
                    result[0] += -0.02244541950900357;
                  }
                } else {
                  result[0] += -0.08738270167947287;
                }
              } else {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                  result[0] += -0.002612532202784162;
                } else {
                  result[0] += 0.00510772437181724;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.154959201812744585) ) ) {
              result[0] += 0.062008845344479716;
            } else {
              result[0] += -0.026971796424327368;
            }
          } else {
            if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
              result[0] += 0.023075377550240486;
            } else {
              result[0] += 0.15418480331758536;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)9.500000000000001776) ) ) {
          if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += -0.0700968632157436;
          } else {
            result[0] += -0.019410105820300987;
          }
        } else {
          if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)10.50000000000000178) ) ) {
            if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += 0.012601433347185604;
            } else {
              result[0] += 0.10299492487609557;
            }
          } else {
            result[0] += -0.03559831992543478;
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.718933820724488193) ) ) {
        result[0] += 0.010400156578384723;
      } else {
        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += -0.08134323183568576;
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.993164777755738193) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)208.0000000000000284) ) ) {
                result[0] += 0.07159375266561786;
              } else {
                result[0] += -0.07891619025124963;
              }
            } else {
              result[0] += -0.0021968924741337815;
            }
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.909855604171753818) ) ) {
            result[0] += 0.011224836336914248;
          } else {
            result[0] += 0.07634270194619285;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)208.0000000000000284) ) ) {
        if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
          result[0] += -0.06844722704794393;
        } else {
          result[0] += 0.007378082017292186;
        }
      } else {
        result[0] += -0.07584158040038874;
      }
    }
  }
  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
    if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
      if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
        if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
              if ( UNLIKELY(  (data[27].missing != -1) && (data[27].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                result[0] += 0.07096206833215932;
              } else {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.255632162094117099) ) ) {
                  result[0] += -0.015325913690693952;
                } else {
                  result[0] += -0.0639117306296104;
                }
              }
            } else {
              if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)6.685099840164185458) ) ) {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)6.117118597030640537) ) ) {
                    if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                      result[0] += -0.0026886579509209007;
                    } else {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.802696108818054643) ) ) {
                        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.97438240051269709) ) ) {
                            result[0] += 0.10886304974987038;
                          } else {
                            result[0] += -0.017146783282867287;
                          }
                        } else {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.41211462020874201) ) ) {
                            result[0] += -0.015993948454008255;
                          } else {
                            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                              result[0] += 0.07993184995116633;
                            } else {
                              result[0] += 0.031770324894644035;
                            }
                          }
                        }
                      } else {
                        result[0] += -0.0022837521922142753;
                      }
                    }
                  } else {
                    result[0] += 0.019423421337730493;
                  }
                } else {
                  result[0] += -0.02974800251045892;
                }
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.801661729812622958) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.48918962478637873) ) ) {
                    result[0] += 0.0043101653749264435;
                  } else {
                    result[0] += 0.029805621211732188;
                  }
                } else {
                  result[0] += 0.03580784028628873;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                  result[0] += -0.10344262645161278;
                } else {
                  result[0] += 0.015319077473114385;
                }
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.232423543930054599) ) ) {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
                    result[0] += 0.0027335027791732837;
                  } else {
                    result[0] += -0.05833799984536338;
                  }
                } else {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                    if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.08240708448822942;
                    } else {
                      result[0] += -0.010217229708970625;
                    }
                  } else {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                        result[0] += -0.052303692299626295;
                      } else {
                        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.142630577087403232) ) ) {
                          result[0] += -0.008449208351089905;
                        } else {
                          result[0] += 0.03037444919077317;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.481347560882569248) ) ) {
                          result[0] += -0.08388028223397818;
                        } else {
                          if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            result[0] += 0.04588610960015768;
                          } else {
                            result[0] += -0.03906970063523754;
                          }
                        }
                      } else {
                        result[0] += -0.05549678258591219;
                      }
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.43742394447326749) ) ) {
                if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                  result[0] += -0.01870279222189943;
                } else {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.09194291443440511;
                  } else {
                    result[0] += -0.08019673145633328;
                  }
                }
              } else {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.142747402191162998) ) ) {
                  result[0] += -0.04650819665567943;
                } else {
                  result[0] += -0.22324116589289936;
                }
              }
            }
          }
        } else {
          result[0] += -0.003325810174913887;
        }
      } else {
        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.938867926597595659) ) ) {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)40.00000000000000711) ) ) {
            if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += 0.02450910802598785;
            } else {
              result[0] += 0.07486888997211524;
            }
          } else {
            result[0] += -0.004654684130637203;
          }
        } else {
          result[0] += 0.004326615352467328;
        }
      }
    } else {
      if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)137422176256.0000153) ) ) {
        if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += -0.001898708355584072;
        } else {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += -0.012836027711000978;
          } else {
            if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
              if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.011180030668329113;
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.255632162094117099) ) ) {
                  result[0] += 0.07523984207376345;
                } else {
                  result[0] += -0.02151298484293993;
                }
              }
            } else {
              result[0] += 0.13692180195576975;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)9.500000000000001776) ) ) {
          if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += -0.06282396618894795;
          } else {
            result[0] += -0.017738396427468488;
          }
        } else {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.357691764831543413) ) ) {
            result[0] += -0.0735613124970108;
          } else {
            if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
              result[0] += -0.003920677220085274;
            } else {
              result[0] += 0.04070603664006648;
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.607751369476319248) ) ) {
        result[0] += 0.007251921327691722;
      } else {
        if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += -0.08344564759422687;
        } else {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
            if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)176.0000000000000284) ) ) {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                  result[0] += 0.0014150892626758462;
                } else {
                  result[0] += 0.11663850993797475;
                }
              } else {
                result[0] += 0.007453510955118334;
              }
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.802901029586792436) ) ) {
                result[0] += 0.10763003505898705;
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.255632162094117099) ) ) {
                  result[0] += -0.009783345060331855;
                } else {
                  if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.924581527709961826) ) ) {
                    result[0] += -0.23575572594219288;
                  } else {
                    result[0] += -0.015698814620722572;
                  }
                }
              }
            }
          } else {
            result[0] += 0.01416835255796539;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)208.0000000000000284) ) ) {
        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
          result[0] += 0.041314560020297215;
        } else {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
            result[0] += -0.0495487987440221;
          } else {
            result[0] += 0.033223899838752;
          }
        }
      } else {
        result[0] += -0.07404727680736084;
      }
    }
  }
  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
    if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
      if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
        if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
            if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)6.685099840164185458) ) ) {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)6.117118597030640537) ) ) {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += -0.002431944087962463;
                } else {
                  if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.10018549361734369;
                  } else {
                    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                      if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += 0.004496399473696455;
                        } else {
                          result[0] += 0.030088957454587506;
                        }
                      } else {
                        result[0] += 0.040948946170108746;
                      }
                    } else {
                      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                        result[0] += -0.002131835592687488;
                      } else {
                        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
                          result[0] += -0.037477596001070215;
                        } else {
                          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.516936540603638583) ) ) {
                            result[0] += 0.11703227172022145;
                          } else {
                            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.993164777755738193) ) ) {
                              result[0] += 0.005266654890064786;
                            } else {
                              result[0] += 0.09136583451059065;
                            }
                          }
                        }
                      }
                    }
                  }
                }
              } else {
                result[0] += 0.019534109267034364;
              }
            } else {
              result[0] += -0.02879430522784915;
            }
          } else {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
              if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.801661729812622958) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.48918962478637873) ) ) {
                      result[0] += 0.004097629503274989;
                    } else {
                      result[0] += 0.02941619969165098;
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.11731815338134943) ) ) {
                      result[0] += -0.06552406232003248;
                    } else {
                      result[0] += 0.03664746818425792;
                    }
                  }
                } else {
                  result[0] += -0.029261799498302012;
                }
              } else {
                if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.019142539242505738;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.108761310577394354) ) ) {
                    result[0] += 0.1284683853028825;
                  } else {
                    result[0] += 0.034950354039220526;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.029068946838379794) ) ) {
                result[0] += -0.08980515241167386;
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
                  result[0] += 0.08873073058730882;
                } else {
                  result[0] += -0.039018441376593965;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
            result[0] += -0.0014073492079188432;
          } else {
            if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.09631291188318265;
              } else {
                result[0] += -0.016503034773161118;
              }
            } else {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += -0.007753272519590998;
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
                  result[0] += 0.03852601767968436;
                } else {
                  result[0] += -0.009365588147139536;
                }
              }
            }
          }
        }
      } else {
        result[0] += 0.006303092443840085;
      }
    } else {
      if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)6.000000000000000888) ) ) {
          if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.605039834976196733) ) ) {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                result[0] += 0.025613172436135364;
              } else {
                result[0] += -0.08088850188980316;
              }
            } else {
              result[0] += -0.0741025339497532;
            }
          } else {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.06896139558830859;
            } else {
              result[0] += 0.016896421259612198;
            }
          }
        } else {
          result[0] += -0.0021466699288355865;
        }
      } else {
        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.255632162094117099) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.843275547027588779) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.599987030029298651) ) ) {
              if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += 0.20028880638102411;
              } else {
                result[0] += 0.008580850099337898;
              }
            } else {
              result[0] += 0.0045389241282090325;
            }
          } else {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.172047138214112216) ) ) {
              result[0] += 0.08597195347479653;
            } else {
              result[0] += -0.0450348396640731;
            }
          }
        } else {
          result[0] += -0.033270656110026615;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.48738741874694913) ) ) {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
        if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.671854496002199042) ) ) {
            if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
              if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += -0.06680367819294263;
              } else {
                result[0] += 0.04722764638167779;
              }
            } else {
              result[0] += 0.05360899293655935;
            }
          } else {
            result[0] += 0.006144475177695505;
          }
        } else {
          result[0] += -0.02839122594393813;
        }
      } else {
        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.450390577316285068) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.778982400894165927) ) ) {
            result[0] += -0.08415588894796447;
          } else {
            result[0] += -0.005581229191645444;
          }
        } else {
          result[0] += -0.25258855520513945;
        }
      }
    } else {
      if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
        if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)257681260544.0000305) ) ) {
          if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)2.087193608283997026) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.431880712509156162) ) ) {
              result[0] += 0.01624795223593009;
            } else {
              result[0] += -0.056318474423818446;
            }
          } else {
            result[0] += 0.1155547208369693;
          }
        } else {
          result[0] += -0.06393720332613696;
        }
      } else {
        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
            result[0] += -0.11440922708800255;
          } else {
            result[0] += 0.01259464376609401;
          }
        } else {
          if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += -0.08344755924038698;
                } else {
                  result[0] += 0.047403241014234744;
                }
              } else {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)176.0000000000000284) ) ) {
                  result[0] += 0.11593355508979629;
                } else {
                  result[0] += 0.014028647562634009;
                }
              }
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += -0.09331062528432431;
              } else {
                result[0] += 0.04287241249479373;
              }
            }
          } else {
            result[0] += 0.07507511468523131;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
    if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
      if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)8816427008.000001907) ) ) {
          result[0] += -0.01835505753970078;
        } else {
          if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
            result[0] += 0.0068987047311643256;
          } else {
            if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)6.685099840164185458) ) ) {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)6.117118597030640537) ) ) {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += -0.002132308784628513;
                } else {
                  if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.09879062641511768;
                  } else {
                    result[0] += 0.005555525760415307;
                  }
                }
              } else {
                result[0] += 0.019701609836846823;
              }
            } else {
              result[0] += -0.028636869128281853;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
          result[0] += 4.7400157157242576e-05;
        } else {
          if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += -0.024320203885499724;
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.605039834976196733) ) ) {
              if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.013848947964434775;
              } else {
                result[0] += -0.06612321531172032;
              }
            } else {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.597323656082154208) ) ) {
                  result[0] += -0.018970710578748697;
                } else {
                  result[0] += 0.057910873942228006;
                }
              } else {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                  result[0] += 0.0021844314208712147;
                } else {
                  result[0] += 0.06432412609886691;
                }
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
        if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += -0.0017822129462134514;
        } else {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.154959201812744585) ) ) {
              result[0] += 0.05915957828614889;
            } else {
              result[0] += -0.024959046356500498;
            }
          } else {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.843275547027588779) ) ) {
              if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.00508368889821173;
                } else {
                  result[0] += 0.07032415282332097;
                }
              } else {
                if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
                  result[0] += 0.2311343611208878;
                } else {
                  result[0] += 0.039484938309854646;
                }
              }
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.329314231872559482) ) ) {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.172047138214112216) ) ) {
                  result[0] += 0.0882794838236485;
                } else {
                  result[0] += -0.031931154217925466;
                }
              } else {
                result[0] += -0.015133322491521974;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)9.500000000000001776) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.851041555404663974) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.285887241363526279) ) ) {
              result[0] += -0.015450759557902958;
            } else {
              result[0] += -0.09425531866731766;
            }
          } else {
            result[0] += -0.06035906483173673;
          }
        } else {
          result[0] += 0.0037313801610467014;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.051912069320679599) ) ) {
      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)176.0000000000000284) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.825115680694581854) ) ) {
          result[0] += 0.04285056269055333;
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.384830474853516513) ) ) {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.718933820724488193) ) ) {
              if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += 0.018603712560533604;
                  } else {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.740319490432739702) ) ) {
                      result[0] += 0.033827270517625914;
                    } else {
                      result[0] += 0.1314338794072424;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.029068946838379794) ) ) {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.02604460716247603) ) ) {
                        result[0] += -0.0529334430551925;
                      } else {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.029068946838379794) ) ) {
                          result[0] += 0.06390281157889365;
                        } else {
                          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                            result[0] += -0.08507753334870517;
                          } else {
                            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.778982400894165927) ) ) {
                              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.694163918495178667) ) ) {
                                result[0] += -0.02432336608679249;
                              } else {
                                result[0] += 0.10016833743870907;
                              }
                            } else {
                              result[0] += -0.06985495079916597;
                            }
                          }
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.484580039978028232) ) ) {
                        result[0] += -0.0822155544338258;
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.484580039978028232) ) ) {
                          result[0] += -0.03884373326145451;
                        } else {
                          result[0] += 0.07711552450537716;
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.302512168884278232) ) ) {
                        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                          result[0] += 0.1710454110130657;
                        } else {
                          result[0] += -0.05302996933822184;
                        }
                      } else {
                        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.607751369476319248) ) ) {
                          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.918272972106934482) ) ) {
                            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.516936540603638583) ) ) {
                              result[0] += -0.07433548445461269;
                            } else {
                              result[0] += 0.07711872527546576;
                            }
                          } else {
                            result[0] += -0.15505828405840538;
                          }
                        } else {
                          result[0] += 0.009686746150070195;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.935600519180298074) ) ) {
                        result[0] += 0.012858713076525436;
                      } else {
                        result[0] += -0.1262939357578191;
                      }
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.302512168884278232) ) ) {
                  result[0] += -0.05973515074409315;
                } else {
                  if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.04675973264182005;
                  } else {
                    if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.918272972106934482) ) ) {
                      result[0] += -0.0011344243982517198;
                    } else {
                      result[0] += 0.1178451051238278;
                    }
                  }
                }
              }
            } else {
              result[0] += 0.03995017348287344;
            }
          } else {
            result[0] += -0.061883915524197856;
          }
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.172047138214112216) ) ) {
          if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
            result[0] += -0.06483355444782535;
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.029068946838379794) ) ) {
              result[0] += -0.05629611498010606;
            } else {
              result[0] += 0.09710547072121606;
            }
          }
        } else {
          result[0] += -0.0673508038056475;
        }
      }
    } else {
      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
        result[0] += -0.027003740789941433;
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.694163918495178667) ) ) {
          result[0] += -0.044930032468013;
        } else {
          result[0] += 0.027760221181034956;
        }
      }
    }
  }
  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)40.00000000000000711) ) ) {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
        if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
          result[0] += 0.015045973917238817;
        } else {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.484580039978028232) ) ) {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += 0.10080323552211097;
            } else {
              result[0] += 0.04017742972752486;
            }
          } else {
            result[0] += -0.03985794929248779;
          }
        }
      } else {
        if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
          if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.76158928871154874) ) ) {
            if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += 0.0009300837946129681;
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.694163918495178667) ) ) {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.041509011767466866;
                } else {
                  if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.09190274051010257;
                  } else {
                    result[0] += 0.0079054984963509;
                  }
                }
              } else {
                result[0] += -0.0262653555136516;
              }
            }
          } else {
            result[0] += 0.017039019383711617;
          }
        } else {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.778982400894165927) ) ) {
            if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.027354496662935247;
              } else {
                result[0] += 0.01849744118875216;
              }
            } else {
              result[0] += -0.027128654644437367;
            }
          } else {
            if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2415.000000000000455) ) ) {
              result[0] += -0.03028920170762346;
            } else {
              result[0] += -0.07926630201161396;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.694163918495178667) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.736135363578796831) ) ) {
          result[0] += -0.005140298495174747;
        } else {
          if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.962127923965454546) ) ) {
              result[0] += -0.030248431511320752;
            } else {
              if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                  result[0] += -0.0041558825123242165;
                } else {
                  result[0] += -0.04747521670886684;
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.1822080612182635) ) ) {
                  result[0] += 0.15498629770008;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.3407330513000506) ) ) {
                    result[0] += -0.06018112287306931;
                  } else {
                    result[0] += 0.02335341637064708;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
              result[0] += 0.020226841118634074;
            } else {
              result[0] += -0.03982300257730128;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.297559976577759233) ) ) {
          result[0] += -0.08468245852043968;
        } else {
          if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
              result[0] += -0.0018381231707047454;
            } else {
              result[0] += -0.04866360011285943;
            }
          } else {
            result[0] += 0.0028056265702512927;
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.56849193572998225) ) ) {
      if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.938867926597595659) ) ) {
        result[0] += -0.08901425410849406;
      } else {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.718933820724488193) ) ) {
          if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.651049375534058505) ) ) {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.029068946838379794) ) ) {
                  result[0] += 0.07009543266207983;
                } else {
                  result[0] += -0.10912527732675162;
                }
              } else {
                if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.589026927947998269) ) ) {
                  result[0] += -0.033812942849770274;
                } else {
                  result[0] += -0.1144053194365567;
                }
              }
            } else {
              if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.142747402191162998) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.778982400894165927) ) ) {
                    result[0] += 0.23056633150528671;
                  } else {
                    result[0] += -0.016608061175338622;
                  }
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.431880712509156162) ) ) {
                    if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                        result[0] += 0.05959716713720978;
                      } else {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.172047138214112216) ) ) {
                          result[0] += 0.019838320367460232;
                        } else {
                          result[0] += -0.14108407848613647;
                        }
                      }
                    } else {
                      result[0] += 0.0604725449583897;
                    }
                  } else {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.617236852645874912) ) ) {
                      result[0] += -0.03965923149179776;
                    } else {
                      result[0] += 0.04584330992727395;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                  if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.589026927947998269) ) ) {
                    if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.778982400894165927) ) ) {
                      result[0] += -0.007168710387720406;
                    } else {
                      result[0] += 0.06722469661427559;
                    }
                  } else {
                    result[0] += 0.11054600629828673;
                  }
                } else {
                  if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.467917680740357333) ) ) {
                    if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += 0.025827566044634973;
                    } else {
                      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.516936540603638583) ) ) {
                        result[0] += -0.02343228418517675;
                      } else {
                        result[0] += -0.1373306606883899;
                      }
                    }
                  } else {
                    result[0] += 0.06965745578725398;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.029068946838379794) ) ) {
                result[0] += 0.04484603338506915;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.516936540603638583) ) ) {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                    result[0] += -0.16797801440766608;
                  } else {
                    result[0] += 0.012580821604714685;
                  }
                } else {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
                      result[0] += 0.10520749600414946;
                    } else {
                      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.607751369476319248) ) ) {
                        result[0] += -0.06061762543805693;
                      } else {
                        result[0] += 0.03864896836957822;
                      }
                    }
                  } else {
                    result[0] += 0.03894974254567667;
                  }
                }
              }
            } else {
              result[0] += -0.06849472975794316;
            }
          }
        } else {
          if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.993164777755738193) ) ) {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)176.0000000000000284) ) ) {
                result[0] += 0.05896835769262612;
              } else {
                result[0] += -0.07646464736748004;
              }
            } else {
              result[0] += -0.013249529670895883;
            }
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)176.0000000000000284) ) ) {
                result[0] += 0.12383238022731369;
              } else {
                result[0] += 0.007126736463378896;
              }
            } else {
              result[0] += 0.020606542935210186;
            }
          }
        }
      }
    } else {
      result[0] += -0.0631642660014569;
    }
  }
  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
    if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
      if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)9.500000000000001776) ) ) {
          result[0] += -0.0012067754713207598;
        } else {
          result[0] += -0.025310735610703905;
        }
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.607751369476319248) ) ) {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.909855604171753818) ) ) {
              if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                result[0] += -0.0037403681861459927;
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.849175214767456943) ) ) {
                  result[0] += -0.03826441699469447;
                } else {
                  result[0] += -0.19153470860319488;
                }
              }
            } else {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                result[0] += -0.07627353808584042;
              } else {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += 0.04689833908130045;
                } else {
                  result[0] += -0.07975232061012605;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.431901693344116655) ) ) {
              result[0] += -0.23433808863324318;
            } else {
              if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)8816427008.000001907) ) ) {
                result[0] += -0.13207178360775682;
              } else {
                if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += -0.010017260469103267;
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
                    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                      result[0] += 0.138759890146526;
                    } else {
                      if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                        result[0] += -0.005084492991983096;
                      } else {
                        result[0] += 0.09635611924439294;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
                      result[0] += -0.0028858138382198235;
                    } else {
                      if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                        result[0] += -0.06908090835364007;
                      } else {
                        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)46.00000000000000711) ) ) {
                          result[0] += -0.056906410574638105;
                        } else {
                          result[0] += 0.0707175671919547;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
            result[0] += 0.04147306573829104;
          } else {
            if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += -0.005338090458281333;
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.357691764831543413) ) ) {
                result[0] += -0.09397887546408157;
              } else {
                if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                  if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.651049375534058505) ) ) {
                    if ( LIKELY(  (data[43].missing != -1) && (data[43].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                      result[0] += 0.0443889813005133;
                    } else {
                      if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                        result[0] += -0.15861465513030681;
                      } else {
                        result[0] += 0.0679171162066571;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.715336322784424716) ) ) {
                      result[0] += 0.09274085004928763;
                    } else {
                      result[0] += -0.02620722162113398;
                    }
                  }
                } else {
                  result[0] += -0.008788873151342066;
                }
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
          if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)6.000000000000000888) ) ) {
            if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.970608234405518466) ) ) {
                if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2727.500000000000455) ) ) {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.699081301689148393) ) ) {
                    result[0] += -0.04956819144224472;
                  } else {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.481121778488159624) ) ) {
                      result[0] += -0.004060420647919563;
                    } else {
                      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                        result[0] += -0.006993102550579823;
                      } else {
                        result[0] += 0.01362536096928011;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
                    result[0] += 0.10954766040436276;
                  } else {
                    result[0] += 0.007234673983390262;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.05094128506367216;
                } else {
                  result[0] += 0.015996073382698768;
                }
              }
            } else {
              if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)6.023992538452149326) ) ) {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.718933820724488193) ) ) {
                  result[0] += 0.04422944101867257;
                } else {
                  result[0] += 0.008767609866152415;
                }
              } else {
                result[0] += -0.04763983925452009;
              }
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.484580039978028232) ) ) {
              if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.198464870452881303) ) ) {
                  result[0] += 0.01734139012148415;
                } else {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.770361423492432529) ) ) {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.060294389724732333) ) ) {
                      if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                        result[0] += 0.02161702649621032;
                      } else {
                        result[0] += -0.09182660553247951;
                      }
                    } else {
                      result[0] += -0.07908464540646715;
                    }
                  } else {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.497866153717041238) ) ) {
                      result[0] += 0.027461822153604484;
                    } else {
                      result[0] += -0.04525395732656061;
                    }
                  }
                }
              } else {
                result[0] += 0.09347642592778764;
              }
            } else {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                result[0] += -0.08052064185190652;
              } else {
                result[0] += -0.023701739904498892;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.445705175399781162) ) ) {
            result[0] += -0.028058813351042163;
          } else {
            if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)4.500000000000000888) ) ) {
              result[0] += 0.01827173858909596;
            } else {
              result[0] += 0.07530884738585418;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.827801465988160068) ) ) {
            if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)4.500000000000000888) ) ) {
              if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)6.023992538452149326) ) ) {
                result[0] += -0.019462851687594544;
              } else {
                result[0] += 0.04225266530354247;
              }
            } else {
              result[0] += 0.016309333731774782;
            }
          } else {
            result[0] += -0.02598266682688102;
          }
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
            result[0] += -0.0035511397605489333;
          } else {
            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
              result[0] += 0.10850784066021264;
            } else {
              result[0] += -0.001093274298723827;
            }
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.284418344497681552) ) ) {
      if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.938867926597595659) ) ) {
        result[0] += -0.08994498408716399;
      } else {
        if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.131699204444885698) ) ) {
            result[0] += -0.06373813350198702;
          } else {
            result[0] += 0.013731815684224845;
          }
        } else {
          result[0] += -0.013574321974707086;
        }
      }
    } else {
      if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
        result[0] += -0.10189165611273672;
      } else {
        result[0] += 0.0007697039062341674;
      }
    }
  }
  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)15.34107780456543146) ) ) {
    if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
      if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
          result[0] += 0.0008521247433208846;
        } else {
          if ( UNLIKELY(  (data[27].missing != -1) && (data[27].fvalue <= (double)-1.00000001800250948e-35) ) ) {
            result[0] += -0.0011951253745926252;
          } else {
            if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.09340371542985271;
            } else {
              result[0] += -0.01835684046463989;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
          if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.993164777755738193) ) ) {
              if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.342454433441162998) ) ) {
                    if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.970608234405518466) ) ) {
                      result[0] += 0.026282062902232697;
                    } else {
                      result[0] += -0.007019541050247838;
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.040716171264650214) ) ) {
                      result[0] += 0.0173270540534505;
                    } else {
                      if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                        result[0] += -0.05499667943057687;
                      } else {
                        result[0] += 0.00653624326892131;
                      }
                    }
                  }
                } else {
                  result[0] += -0.03253888447261621;
                }
              } else {
                if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.04636812210083185) ) ) {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
                        result[0] += -0.028027032405319126;
                      } else {
                        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.094205617904663974) ) ) {
                          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.827801465988160068) ) ) {
                            if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                              result[0] += -0.002614978493095639;
                            } else {
                              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.673553824424744096) ) ) {
                                result[0] += -0.0020985551500730094;
                              } else {
                                result[0] += 0.07859640440866972;
                              }
                            }
                          } else {
                            result[0] += -0.07804034817901212;
                          }
                        } else {
                          result[0] += -0.021672334351319444;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.302512168884278232) ) ) {
                        result[0] += -0.008798857053590547;
                      } else {
                        result[0] += 0.04091598874953935;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += -0.0037509310996647555;
                    } else {
                      result[0] += 0.06697961576023496;
                    }
                  }
                } else {
                  result[0] += -0.04621833540647895;
                }
              }
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
                result[0] += -0.05012986857242621;
              } else {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                  result[0] += -0.04000732220796337;
                } else {
                  result[0] += 0.01910209224660124;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += 0.08494014990486098;
              } else {
                result[0] += -0.009162214232604915;
              }
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.95478391647339045) ) ) {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
                  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.284418344497681552) ) ) {
                      result[0] += 0.01305685389741005;
                    } else {
                      result[0] += -0.06800872585581351;
                    }
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
                      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.484580039978028232) ) ) {
                          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                            result[0] += 0.04434571044109227;
                          } else {
                            result[0] += -0.0013010123968249013;
                          }
                        } else {
                          result[0] += 0.04913942706432513;
                        }
                      } else {
                        result[0] += -0.06356326612848154;
                      }
                    } else {
                      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.327068090438843662) ) ) {
                          result[0] += 0.0029030780578149177;
                        } else {
                          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                            if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
                              result[0] += -0.09581173938444695;
                            } else {
                              result[0] += -0.025054908060966302;
                            }
                          } else {
                            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                              result[0] += -0.09557050048141863;
                            } else {
                              result[0] += 0.07556284296003962;
                            }
                          }
                        }
                      } else {
                        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.607751369476319248) ) ) {
                          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                            if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                              result[0] += -0.04812636297365652;
                            } else {
                              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.284418344497681552) ) ) {
                                result[0] += 0.0117752197208466;
                              } else {
                                result[0] += 0.11996770821189624;
                              }
                            }
                          } else {
                            result[0] += -0.061562788086215584;
                          }
                        } else {
                          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.993164777755738193) ) ) {
                            if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
                              result[0] += -0.04738379874708925;
                            } else {
                              result[0] += -0.1826264590253651;
                            }
                          } else {
                            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.543407917022706854) ) ) {
                              result[0] += -0.08195558126576973;
                            } else {
                              result[0] += 0.06878610023496282;
                            }
                          }
                        }
                      }
                    }
                  }
                } else {
                  result[0] += -0.04291034297402724;
                }
              } else {
                result[0] += -0.06617697421103556;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += 0.006583666120075476;
          } else {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.516936540603638583) ) ) {
              if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.06902920623609114;
                } else {
                  result[0] += 0.0011599485760363094;
                }
              } else {
                result[0] += 0.001388844033181616;
              }
            } else {
              result[0] += 0.011280567661333505;
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)1.497866153717041238) ) ) {
          if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += -0.009800242385045413;
          } else {
            result[0] += -0.10512627737375389;
          }
        } else {
          result[0] += -0.0005000404165311837;
        }
      } else {
        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)9.500000000000001776) ) ) {
          result[0] += -0.02463022200611723;
        } else {
          if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.994492053985595925) ) ) {
              result[0] += 0.007437148130309991;
            } else {
              result[0] += -0.05638894257431109;
            }
          } else {
            result[0] += 0.023225314730002958;
          }
        }
      }
    }
  } else {
    if ( UNLIKELY(  (data[27].missing != -1) && (data[27].fvalue <= (double)-1.00000001800250948e-35) ) ) {
      result[0] += -0.15095767545249783;
    } else {
      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
        result[0] += 0.06820718812577704;
      } else {
        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
          result[0] += 0.020707477117834446;
        } else {
          result[0] += -0.03298608956801801;
        }
      }
    }
  }
  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)15.34107780456543146) ) ) {
    if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
      if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
        if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)21466447872.00000381) ) ) {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.607737541198732245) ) ) {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.238486170768738237) ) ) {
                    if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                      result[0] += -0.01042993822013863;
                    } else {
                      if ( UNLIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += 0.016269960357140928;
                      } else {
                        result[0] += -0.005078834285200234;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)8816427008.000001907) ) ) {
                        result[0] += 0.08507897887567314;
                      } else {
                        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.134366035461426669) ) ) {
                          result[0] += -0.009317377660599214;
                        } else {
                          if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
                            if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.589026927947998269) ) ) {
                              result[0] += 0.019035508862741548;
                            } else {
                              result[0] += 0.07533549556405386;
                            }
                          } else {
                            result[0] += -0.05201277643984225;
                          }
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += -0.005490548070022016;
                      } else {
                        result[0] += 0.0085587114246334;
                      }
                    }
                  }
                } else {
                  result[0] += 0.06582536099386564;
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.484580039978028232) ) ) {
                  result[0] += 0.0022330341207338874;
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.094205617904663974) ) ) {
                    result[0] += -0.051324112839406855;
                  } else {
                    result[0] += -0.009129817254108422;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.433569431304932529) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.094205617904663974) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.142630577087403232) ) ) {
                      result[0] += -0.013044771820787721;
                    } else {
                      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                        result[0] += 0.08065597069146804;
                      } else {
                        result[0] += 0.011532980130407874;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                        if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.589026927947998269) ) ) {
                          result[0] += -0.02376907730886291;
                        } else {
                          result[0] += -0.07933881646074492;
                        }
                      } else {
                        result[0] += 0.003707734523327492;
                      }
                    } else {
                      if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2415.000000000000455) ) ) {
                        if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += 0.1316707249318312;
                        } else {
                          if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                              result[0] += -0.08031757546979253;
                            } else {
                              result[0] += 0.026610392202484817;
                            }
                          } else {
                            result[0] += 0.09823023037820088;
                          }
                        }
                      } else {
                        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                          result[0] += -0.10249123780154622;
                        } else {
                          result[0] += 0.06634091135432207;
                        }
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
                      result[0] += 0.0004706865062907939;
                    } else {
                      result[0] += 0.09304576217103441;
                    }
                  } else {
                    result[0] += 0.011292131129297667;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.467917680740357333) ) ) {
                  result[0] += -0.03866936685969725;
                } else {
                  if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += -0.007755976764516505;
                  } else {
                    if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.85341548919677912) ) ) {
                        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.276966691017151323) ) ) {
                          if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                            result[0] += -0.010004279880738709;
                          } else {
                            result[0] += 0.06475526815739217;
                          }
                        } else {
                          result[0] += 0.08636675214098288;
                        }
                      } else {
                        result[0] += -0.035667304348757776;
                      }
                    } else {
                      result[0] += 0.039073914849827335;
                    }
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.285887241363526279) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.422362327575684482) ) ) {
                  result[0] += -0.001398280009454615;
                } else {
                  result[0] += 0.03387220090627349;
                }
              } else {
                result[0] += -0.031739771371715436;
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.94957673549652144) ) ) {
                result[0] += -0.055834160369920255;
              } else {
                if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                  result[0] += -0.04150227390474863;
                } else {
                  result[0] += -0.007254745286524373;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.605039834976196733) ) ) {
              if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += -0.0048053446907768945;
              } else {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
                  result[0] += -0.01935605283488046;
                } else {
                  result[0] += 0.05547138681845885;
                }
              }
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.255632162094117099) ) ) {
                result[0] += -0.012912557399491898;
              } else {
                result[0] += -0.05033122095011966;
              }
            }
          } else {
            result[0] += 0.005341705831205129;
          }
        }
      } else {
        if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)137422176256.0000153) ) ) {
          result[0] += -0.0006919027850904216;
        } else {
          if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)9.500000000000001776) ) ) {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
              result[0] += -0.011305162052406906;
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.284418344497681552) ) ) {
                result[0] += -0.010093137875722826;
              } else {
                result[0] += -0.05456670356940782;
              }
            }
          } else {
            if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.284418344497681552) ) ) {
                result[0] += -0.0034115553804811762;
              } else {
                result[0] += -0.06996883588588822;
              }
            } else {
              result[0] += 0.02128448430551981;
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.45958471298217951) ) ) {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.154959201812744585) ) ) {
          result[0] += 0.053710597628046246;
        } else {
          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
            result[0] += -0.02992512732295237;
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.481347560882569248) ) ) {
              result[0] += 0.06291475244761471;
            } else {
              result[0] += -0.02406562423928931;
            }
          }
        }
      } else {
        result[0] += -0.10011070049397666;
      }
    }
  } else {
    if ( UNLIKELY(  (data[27].missing != -1) && (data[27].fvalue <= (double)-1.00000001800250948e-35) ) ) {
      result[0] += -0.14231980837946315;
    } else {
      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
        result[0] += 0.059426214855628295;
      } else {
        result[0] += -0.018028478093836744;
      }
    }
  }
  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)15.34107780456543146) ) ) {
    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
      if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)24.00000000000000355) ) ) {
        if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)4.500000000000000888) ) ) {
          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
              result[0] += -0.0011334547195375267;
            } else {
              result[0] += 0.006528133989274672;
            }
          } else {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
                  if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.2687106132507342) ) ) {
                      result[0] += 0.005439085119280916;
                    } else {
                      result[0] += 0.04463841677198094;
                    }
                  } else {
                    if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += 0.006066003500939027;
                    } else {
                      result[0] += -0.031654001631308895;
                    }
                  }
                } else {
                  result[0] += -0.05222781430793608;
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.827801465988160068) ) ) {
                  if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                    result[0] += 0.021953320611669883;
                  } else {
                    result[0] += -0.013843789782473302;
                  }
                } else {
                  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)4.500000000000000888) ) ) {
                    result[0] += 0.022114002963935602;
                  } else {
                    if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                      result[0] += -0.0038084712771970647;
                    } else {
                      result[0] += 0.0662445372126229;
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.827801465988160068) ) ) {
                  result[0] += 0.0013529741482309916;
                } else {
                  result[0] += -0.02481494711338775;
                }
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                  result[0] += -0.005886474876328602;
                } else {
                  result[0] += 0.07363071407723668;
                }
              }
            }
          }
        } else {
          result[0] += 0.07079091283026766;
        }
      } else {
        result[0] += -0.08493298284319682;
      }
    } else {
      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.667095184326172763) ) ) {
        if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.938867926597595659) ) ) {
          result[0] += -0.09122522302274583;
        } else {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.718933820724488193) ) ) {
            if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.534971714019776279) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.029068946838379794) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.25437736511230646) ) ) {
                      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                        result[0] += -0.06798339427381427;
                      } else {
                        result[0] += 0.09901322941017086;
                      }
                    } else {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.284418344497681552) ) ) {
                        result[0] += -0.04348709575249518;
                      } else {
                        result[0] += 0.0641339553094389;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.158952236175537998) ) ) {
                        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                          result[0] += -0.0686921033301127;
                        } else {
                          result[0] += 0.03826441277818309;
                        }
                      } else {
                        result[0] += 0.05539594409042553;
                      }
                    } else {
                      if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.158761024475098544) ) ) {
                        result[0] += -0.05392374291150684;
                      } else {
                        result[0] += -0.1914904070542064;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.431880712509156162) ) ) {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.161602735519410068) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.715336322784424716) ) ) {
                        result[0] += 0.06628339388042565;
                      } else {
                        result[0] += -0.016927547281617827;
                      }
                    } else {
                      result[0] += 0.04115148582563573;
                    }
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.445957899093628818) ) ) {
                      result[0] += -0.23851963503278362;
                    } else {
                      result[0] += -0.024419178745109613;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.802696108818054643) ) ) {
                  result[0] += -0.05430621620098929;
                } else {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                    if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)4.500000000000000888) ) ) {
                      result[0] += 0.055667988743561436;
                    } else {
                      result[0] += -0.048343644873207296;
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.135017871856690341) ) ) {
                      result[0] += 0.14567248918092265;
                    } else {
                      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                        result[0] += 0.02897481954783084;
                      } else {
                        result[0] += -0.031281913286687536;
                      }
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.516936540603638583) ) ) {
                  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += 0.061771568462717176;
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.827801465988160068) ) ) {
                      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                        result[0] += 0.061467946974609514;
                      } else {
                        result[0] += -0.022071868875652302;
                      }
                    } else {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.85341548919677912) ) ) {
                        result[0] += -0.007255340523523087;
                      } else {
                        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.918272972106934482) ) ) {
                            result[0] += -0.08242042441928837;
                          } else {
                            result[0] += -0.18326164941695308;
                          }
                        } else {
                          result[0] += 0.05274639219978945;
                        }
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.208071470260621005) ) ) {
                      result[0] += 0.034481080820661036;
                    } else {
                      result[0] += -0.06519314339762423;
                    }
                  } else {
                    result[0] += 0.05303497102412045;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                  if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.013672759022096501;
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.172047138214112216) ) ) {
                      result[0] += -0.03286033073902747;
                    } else {
                      result[0] += 1.3724150598941205;
                    }
                  }
                } else {
                  result[0] += -0.09840540647612035;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.141444921493531162) ) ) {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)176.0000000000000284) ) ) {
                  result[0] += 0.04205003671480781;
                } else {
                  result[0] += -0.06874974628393556;
                }
              } else {
                result[0] += -0.016485520575816797;
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.909855604171753818) ) ) {
                result[0] += -0.01534712333165813;
              } else {
                result[0] += 0.06563262181264605;
              }
            }
          }
        }
      } else {
        result[0] += 0.15379640942371386;
      }
    }
  } else {
    if ( UNLIKELY(  (data[27].missing != -1) && (data[27].fvalue <= (double)-1.00000001800250948e-35) ) ) {
      result[0] += -0.14533846863463315;
    } else {
      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
        if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
          result[0] += 0.002640602238220906;
        } else {
          result[0] += 0.11638718839224907;
        }
      } else {
        result[0] += -0.01879035024034235;
      }
    }
  }
}

