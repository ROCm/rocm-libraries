
#include "header.h"

void predict_unit2(union Entry* data, double* result) {
  unsigned int tmp;
  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)20.00000000000000355) ) ) {
    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.65882921218872248) ) ) {
        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)120.0000000000000142) ) ) {
          result[0] += 0.0010767602844746196;
        } else {
          result[0] += -0.04402525884942765;
        }
      } else {
        if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)8.500000000000001776) ) ) {
          if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)6.500000000000000888) ) ) {
            result[0] += 0.01775040118416212;
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.02604460716247603) ) ) {
              result[0] += -0.18683151096659853;
            } else {
              result[0] += 0.04854853095754403;
            }
          }
        } else {
          if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)25.50000000000000355) ) ) {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.042435407638550693) ) ) {
              result[0] += 0.04469953365582003;
            } else {
              result[0] += -0.07845125178392989;
            }
          } else {
            result[0] += 0.01749023455280798;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
        if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.938867926597595659) ) ) {
          result[0] += 0.04208458395237915;
        } else {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.90474271774292081) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.246492147445679599) ) ) {
              result[0] += -0.08994734819184556;
            } else {
              result[0] += 0.21432426982102376;
            }
          } else {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
              if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)7.500000000000000888) ) ) {
                if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)10.50000000000000178) ) ) {
                  result[0] += 0.15325839440498457;
                } else {
                  result[0] += -0.0830247924908577;
                }
              } else {
                result[0] += 0.2418950452264754;
              }
            } else {
              result[0] += -0.0637709212736349;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.894675970077515537) ) ) {
          result[0] += -0.06890751080039192;
        } else {
          if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)5.500000000000000888) ) ) {
            if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.500000000000000444) ) ) {
              if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)5.500000000000000888) ) ) {
                  if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)4.500000000000000888) ) ) {
                    result[0] += -0.007610044494201697;
                  } else {
                    if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)9.500000000000001776) ) ) {
                      result[0] += -0.07128303788936127;
                    } else {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                        result[0] += -0.02320716140444888;
                      } else {
                        result[0] += 0.06728067595003792;
                      }
                    }
                  }
                } else {
                  result[0] += -0.065544495561186;
                }
              } else {
                result[0] += -0.05742532429771313;
              }
            } else {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)120.0000000000000142) ) ) {
                if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.718933820724488193) ) ) {
                    if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                      result[0] += -0.05832023546641546;
                    } else {
                      result[0] += 0.010871529157615353;
                    }
                  } else {
                    result[0] += -0.07203462189612059;
                  }
                } else {
                  if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.827801465988160068) ) ) {
                    result[0] += 0.028750240532816725;
                  } else {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += -0.10026543720338642;
                    } else {
                      result[0] += 0.05253518066376997;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.861792564392090288) ) ) {
                  result[0] += -0.05369503329804224;
                } else {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                    if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                      result[0] += 0.011751723016600348;
                    } else {
                      result[0] += -0.05241185680255207;
                    }
                  } else {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += 0.045604596405317555;
                    } else {
                      if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += 0.02577454136408591;
                      } else {
                        result[0] += -0.1048056019593868;
                      }
                    }
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
              if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)21.50000000000000355) ) ) {
                if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)15.50000000000000178) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                      result[0] += 0.11627815305795647;
                    } else {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.993164777755738193) ) ) {
                        if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += -0.031629295279502076;
                        } else {
                          result[0] += -0.14869494072993655;
                        }
                      } else {
                        result[0] += -0.007523592914260448;
                      }
                    }
                  } else {
                    result[0] += 0.011008794127889888;
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.778982400894165927) ) ) {
                    result[0] += 0.03386413722241474;
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.827801465988160068) ) ) {
                      result[0] += 0.06683105870782301;
                    } else {
                      result[0] += -0.08862386032576117;
                    }
                  }
                }
              } else {
                result[0] += 0.031555119976093345;
              }
            } else {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.154959201812744585) ) ) {
                  result[0] += 0.06518275798077271;
                } else {
                  result[0] += -0.04615209922949725;
                }
              } else {
                result[0] += 0.02660782678621587;
              }
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)40.00000000000000711) ) ) {
      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.827801465988160068) ) ) {
        result[0] += -0.020598952847658822;
      } else {
        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)56.00000000000000711) ) ) {
          result[0] += -0.09548352124170617;
        } else {
          result[0] += -0.019884174258295;
        }
      }
    } else {
      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.418141007423401323) ) ) {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.764703154563904253) ) ) {
                result[0] += -0.05263742649013681;
              } else {
                result[0] += 0.12189788845319614;
              }
            } else {
              result[0] += 0.012870170540519586;
            }
          } else {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
              result[0] += -0.08775939494138256;
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.481121778488159624) ) ) {
                result[0] += 0.022522778422326452;
              } else {
                result[0] += -0.034645340659289946;
              }
            }
          }
        } else {
          result[0] += 0.02179544923216095;
        }
      } else {
        if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
          result[0] += -0.04111844123058933;
        } else {
          if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)2.500000000000000444) ) ) {
            result[0] += 0.024049163819692625;
          } else {
            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
              if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)7.000000000000000888) ) ) {
                result[0] += -0.0012664062092665482;
              } else {
                if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.607751369476319248) ) ) {
                    result[0] += 0.0007615121453403164;
                  } else {
                    result[0] += -0.03051464572952578;
                  }
                } else {
                  result[0] += -0.10343303301254113;
                }
              }
            } else {
              result[0] += -0.04502045907788528;
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)20.00000000000000355) ) ) {
    if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
      if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.284418344497681552) ) ) {
          result[0] += 0.004176086991616923;
        } else {
          result[0] += 0.03112693379711869;
        }
      } else {
        if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)7.500000000000000888) ) ) {
          if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)5.500000000000000888) ) ) {
              if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)10.50000000000000178) ) ) {
                if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)9.500000000000001776) ) ) {
                  if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)4.500000000000000888) ) ) {
                    result[0] += -0.002881808139001871;
                  } else {
                    result[0] += -0.07764249812389959;
                  }
                } else {
                  result[0] += 0.03964045948678077;
                }
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.651049375534058505) ) ) {
                  if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.06261888354187749;
                  } else {
                    result[0] += 0.029996397960190897;
                  }
                } else {
                  result[0] += -0.07355511852170848;
                }
              }
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.651049375534058505) ) ) {
                if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)6.500000000000000888) ) ) {
                  if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)9.500000000000001776) ) ) {
                    result[0] += 0.08248822547469964;
                  } else {
                    result[0] += 0.023625969882128957;
                  }
                } else {
                  result[0] += -0.03905697587888939;
                }
              } else {
                if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)6.500000000000000888) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.142630577087403232) ) ) {
                    if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)18.50000000000000355) ) ) {
                      result[0] += -0.033606357586176146;
                    } else {
                      result[0] += 0.13716502067094125;
                    }
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.051912069320679599) ) ) {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.849175214767456943) ) ) {
                        result[0] += 0.0020228642473166464;
                      } else {
                        result[0] += -0.07021333144140253;
                      }
                    } else {
                      result[0] += -0.0904140582928438;
                    }
                  }
                } else {
                  result[0] += 0.018531373211906774;
                }
              }
            }
          } else {
            result[0] += -0.050336683047941644;
          }
        } else {
          if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
              result[0] += -0.07118614246375791;
            } else {
              result[0] += -0.006847720573807107;
            }
          } else {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)120.0000000000000142) ) ) {
              if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)112.0000000000000142) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.778982400894165927) ) ) {
                  if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.284418344497681552) ) ) {
                    result[0] += 0.05388138221887659;
                  } else {
                    result[0] += 0.012878890882049238;
                  }
                } else {
                  if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.718933820724488193) ) ) {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.484580039978028232) ) ) {
                        result[0] += 0.04764456818183312;
                      } else {
                        result[0] += -0.04243147959457577;
                      }
                    } else {
                      result[0] += 0.008091212349508221;
                    }
                  } else {
                    result[0] += -0.06650817220791715;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.142630577087403232) ) ) {
                  result[0] += -0.05530890103279277;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.456172943115236151) ) ) {
                    result[0] += -0.03585027305697717;
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.02604460716247603) ) ) {
                      result[0] += -0.1890916888098657;
                    } else {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.918272972106934482) ) ) {
                        result[0] += 0.0664565698551806;
                      } else {
                        result[0] += 0.028240697621380352;
                      }
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.918272972106934482) ) ) {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
                  result[0] += -0.05980019875380871;
                } else {
                  if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.172047138214112216) ) ) {
                      if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                        result[0] += -0.027349912786061078;
                      } else {
                        result[0] += -0.12342374441067992;
                      }
                    } else {
                      result[0] += 0.054901810002535294;
                    }
                  } else {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.607751369476319248) ) ) {
                      result[0] += -0.009596622153804737;
                    } else {
                      result[0] += 0.05224876463425219;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.651049375534058505) ) ) {
                    result[0] += 0.05510376536819592;
                  } else {
                    result[0] += -0.07794851085191327;
                  }
                } else {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.624251961708069292) ) ) {
                    result[0] += -0.060001719439324634;
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.029068946838379794) ) ) {
                      result[0] += -0.005765445987973861;
                    } else {
                      if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)232.0000000000000284) ) ) {
                          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                            result[0] += 0.021471292373083904;
                          } else {
                            result[0] += -0.04454846551041783;
                          }
                        } else {
                          result[0] += 0.07913838398636003;
                        }
                      } else {
                        result[0] += 0.09089701806428929;
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
      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.802696108818054643) ) ) {
        result[0] += -0.0374314036489272;
      } else {
        if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)2.500000000000000444) ) ) {
          result[0] += 0.053448692892128315;
        } else {
          if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)16.50000000000000355) ) ) {
            result[0] += 0.01952521451551477;
          } else {
            result[0] += -0.05851680622115957;
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)40.00000000000000711) ) ) {
      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.418141007423401323) ) ) {
        result[0] += -0.010505940527407373;
      } else {
        if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
          result[0] += -0.0869672789408119;
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.938988685607911933) ) ) {
            result[0] += 0.01694212780158697;
          } else {
            if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
              if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)4.500000000000000888) ) ) {
                result[0] += -0.03759026707674118;
              } else {
                result[0] += 0.021003047259991293;
              }
            } else {
              result[0] += 0.022196961996682293;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.736135363578796831) ) ) {
        result[0] += 0.011798723900357196;
      } else {
        if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
            result[0] += -0.021077651287258938;
          } else {
            result[0] += -0.06485800717691914;
          }
        } else {
          if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.500000000000000444) ) ) {
            result[0] += 0.02147184665905067;
          } else {
            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)80.00000000000001421) ) ) {
                result[0] += -0.026615871964336714;
              } else {
                result[0] += 0.0023026416519697497;
              }
            } else {
              if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)6.117235183715821201) ) ) {
                result[0] += -0.049566299402061885;
              } else {
                result[0] += 0.05021327128462504;
              }
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)20.00000000000000355) ) ) {
    if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)17.50000000000000355) ) ) {
      if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.04969787597656428) ) ) {
          if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)7.500000000000000888) ) ) {
            result[0] += 0.0019262198367309526;
          } else {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.012675821781158891) ) ) {
              result[0] += -0.12405248566301284;
            } else {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += -0.038788543013692445;
              } else {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                  result[0] += 0.03144808148740132;
                } else {
                  result[0] += -0.1406031959845162;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)8.500000000000001776) ) ) {
            result[0] += 0.017040998796618734;
          } else {
            result[0] += -0.015034682429094707;
          }
        }
      } else {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.284998416900635654) ) ) {
            result[0] += 0.035213564122830596;
          } else {
            result[0] += -0.08658926250692389;
          }
        } else {
          if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)29.50000000000000355) ) ) {
            if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)8.500000000000001776) ) ) {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.516936540603638583) ) ) {
                  if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.909855604171753818) ) ) {
                      result[0] += -0.08013646933124687;
                    } else {
                      result[0] += 0.10805019475446809;
                    }
                  } else {
                    result[0] += -0.06909320931908869;
                  }
                } else {
                  result[0] += 0.01859694000447624;
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.284998416900635654) ) ) {
                  result[0] += -0.08137893737484807;
                } else {
                  if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.500000000000000444) ) ) {
                    if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)9.500000000000001776) ) ) {
                      if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += -0.006968339482579787;
                      } else {
                        result[0] += -0.053791468604885144;
                      }
                    } else {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.651049375534058505) ) ) {
                        if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)9.500000000000001776) ) ) {
                          result[0] += 0.062052618676830165;
                        } else {
                          result[0] += 0.016707916522406958;
                        }
                      } else {
                        if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)10.50000000000000178) ) ) {
                          result[0] += 0.05022822155238648;
                        } else {
                          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.802696108818054643) ) ) {
                            result[0] += 0.0702842549253112;
                          } else {
                            result[0] += -0.06500194920456619;
                          }
                        }
                      }
                    }
                  } else {
                    result[0] += 0.010442523968737889;
                  }
                }
              }
            } else {
              result[0] += -0.04491375988496798;
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.993164777755738193) ) ) {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.384830474853516513) ) ) {
                  result[0] += -0.16406682915529605;
                } else {
                  result[0] += 0.14892288064595846;
                }
              } else {
                result[0] += 0.024434168857826025;
              }
            } else {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += 0.13236938477081692;
              } else {
                result[0] += 0.03049908710342893;
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.484580039978028232) ) ) {
        if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
          result[0] += 0.01803907976920124;
        } else {
          result[0] += -0.0644022731205468;
        }
      } else {
        if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)6.500000000000000888) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.651049375534058505) ) ) {
            result[0] += 0.1189317332894897;
          } else {
            result[0] += -0.102447612485817;
          }
        } else {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.342454433441162998) ) ) {
            result[0] += -0.010454875895635459;
          } else {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.780892848968506748) ) ) {
                  result[0] += 0.056449082272494434;
                } else {
                  result[0] += -0.027740213994204524;
                }
              } else {
                result[0] += 0.0041084686844899115;
              }
            } else {
              if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += 0.06425994982570081;
              } else {
                result[0] += 0.01268997848960806;
              }
            }
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)40.00000000000000711) ) ) {
      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)6.561806440353394443) ) ) {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.515218973159790483) ) ) {
            result[0] += 0.06530388072437814;
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += -0.06685282362573806;
              } else {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.039720773696899636) ) ) {
                  result[0] += 0.01504413082331396;
                } else {
                  result[0] += 0.0883253476985893;
                }
              }
            } else {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.172047138214112216) ) ) {
                result[0] += -0.0152513796517154;
              } else {
                result[0] += -0.11323191835353166;
              }
            }
          }
        } else {
          result[0] += 0.09924617614238661;
        }
      } else {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.41262340545654475) ) ) {
              result[0] += 0.07915917965458853;
            } else {
              if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
                result[0] += 0.09014600817919038;
              } else {
                result[0] += -0.07026113433877841;
              }
            }
          } else {
            result[0] += -0.07100109872928338;
          }
        } else {
          result[0] += -0.008064140141004067;
        }
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.736135363578796831) ) ) {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.827801465988160068) ) ) {
              result[0] += -0.011479706741768176;
            } else {
              if ( UNLIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += 0.0970652194102787;
              } else {
                result[0] += 0.003116338007503881;
              }
            }
          } else {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.524927973747253862) ) ) {
              result[0] += 0.0195657945859936;
            } else {
              result[0] += -0.04215237947174868;
            }
          }
        } else {
          result[0] += 0.029450586154460036;
        }
      } else {
        if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
          result[0] += -0.03692223933978059;
        } else {
          if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.500000000000000444) ) ) {
            result[0] += 0.019104553083809258;
          } else {
            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += 0.0014337654310281412;
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.48918962478637873) ) ) {
                  if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                    result[0] += 0.04314980978864903;
                  } else {
                    result[0] += -0.0912728189919193;
                  }
                } else {
                  result[0] += 0.07398614201664815;
                }
              }
            } else {
              result[0] += -0.039319608861252055;
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)5.500000000000000888) ) ) {
    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.79870843887329279) ) ) {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
              if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)10.50000000000000178) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.994492053985595925) ) ) {
                  result[0] += 0.00219697642101476;
                } else {
                  if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                    if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)6.500000000000000888) ) ) {
                      result[0] += 0.01620081747877048;
                    } else {
                      result[0] += 0.06457437129843734;
                    }
                  } else {
                    result[0] += 0.06480358218153469;
                  }
                }
              } else {
                result[0] += -0.03793791974322505;
              }
            } else {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.03984264490456303;
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.580392837524414951) ) ) {
                  result[0] += -0.014718826093391542;
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.516936540603638583) ) ) {
                    result[0] += -0.3733517179442952;
                  } else {
                    result[0] += 0.007870555334560199;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.0829800624467896;
            } else {
              if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                result[0] += 0.02664214655028941;
              } else {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)80.00000000000001421) ) ) {
                  if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.051912069320679599) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.484580039978028232) ) ) {
                      result[0] += 0.004487675799433042;
                    } else {
                      result[0] += -0.022350691298815333;
                    }
                  } else {
                    result[0] += -0.07900557799481364;
                  }
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.232423543930054599) ) ) {
                    result[0] += -0.06664128433025722;
                  } else {
                    result[0] += 0.017949879461342643;
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += 0.011949793361627313;
          } else {
            result[0] += 0.05925808334525467;
          }
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.02604460716247603) ) ) {
          result[0] += -0.11060484405338566;
        } else {
          result[0] += 0.045641118824849364;
        }
      }
    } else {
      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.516936540603638583) ) ) {
          if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.10290670394897639) ) ) {
              result[0] += 0.07211618311791389;
            } else {
              result[0] += -0.05728323482227694;
            }
          } else {
            result[0] += -0.09060318176021656;
          }
        } else {
          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.516936540603638583) ) ) {
              result[0] += -0.09513809598292258;
            } else {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)72.00000000000001421) ) ) {
                if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
                  result[0] += -0.0821339079520378;
                } else {
                  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
                    result[0] += 0.01720327560430374;
                  } else {
                    result[0] += -0.09919748597983145;
                  }
                }
              } else {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.088880300521851474) ) ) {
                  result[0] += -0.06713282195388574;
                } else {
                  result[0] += 0.056909167603602845;
                }
              }
            }
          } else {
            result[0] += 0.04257114594078854;
          }
        }
      } else {
        if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.475347042083742011) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.894675970077515537) ) ) {
                result[0] += -0.05541688946084059;
              } else {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += 0.006876143690423055;
                } else {
                  result[0] += 0.056458041308992395;
                }
              }
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.778982400894165927) ) ) {
                if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.05853260357128939;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.39281225204467951) ) ) {
                    if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
                      result[0] += 0.003845627617786495;
                    } else {
                      result[0] += 0.13568008968561376;
                    }
                  } else {
                    result[0] += -0.03835606000423166;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                    if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)7.500000000000000888) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.795762062072754794) ) ) {
                        result[0] += 0.014165934141971412;
                      } else {
                        result[0] += 0.10325369338859423;
                      }
                    } else {
                      result[0] += -0.07748725530129079;
                    }
                  } else {
                    if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)4.500000000000000888) ) ) {
                      result[0] += -0.01177831045322044;
                    } else {
                      result[0] += 0.045442879005713574;
                    }
                  }
                } else {
                  result[0] += -0.05331948274257327;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.484580039978028232) ) ) {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.7512402534484881) ) ) {
                  result[0] += 0.019333094890879306;
                } else {
                  result[0] += -0.017654311590003174;
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.671854496002199042) ) ) {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += 0.0011610479199873258;
                  } else {
                    result[0] += -0.08892650695430288;
                  }
                } else {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                    result[0] += 0.014515285502834794;
                  } else {
                    result[0] += 0.04595075376126263;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                  result[0] += 0.02759288521119252;
                } else {
                  result[0] += -0.1004084215853436;
                }
              } else {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)6.500000000000000888) ) ) {
                        result[0] += -0.018794045972319942;
                      } else {
                        result[0] += 0.025130356481151778;
                      }
                    } else {
                      if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += -0.17573909886489472;
                      } else {
                        result[0] += 0.07567631331438435;
                      }
                    }
                  } else {
                    result[0] += -0.09177208091729826;
                  }
                } else {
                  if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
                    result[0] += -0.042697320644632;
                  } else {
                    result[0] += 0.11694600220455663;
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.094205617904663974) ) ) {
              result[0] += -0.07128436755363568;
            } else {
              result[0] += 0.00025115209412916945;
            }
          } else {
            result[0] += 0.012060252282598568;
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)12.50000000000000178) ) ) {
      result[0] += 0.015032266281703994;
    } else {
      result[0] += -0.0629693450688251;
    }
  }
  if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)5.500000000000000888) ) ) {
    if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.79870843887329279) ) ) {
          if ( UNLIKELY(  (data[29].missing != -1) && (data[29].fvalue <= (double)-1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.215408444404602495) ) ) {
                result[0] += 0.0018716004593522268;
              } else {
                if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)10.50000000000000178) ) ) {
                  if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                    if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)6.500000000000000888) ) ) {
                      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                        if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)2.500000000000000444) ) ) {
                          result[0] += 0.02808668638075908;
                        } else {
                          result[0] += -0.04782008215218219;
                        }
                      } else {
                        result[0] += -0.034870859410792;
                      }
                    } else {
                      result[0] += 0.06096393597569249;
                    }
                  } else {
                    result[0] += 0.06602931522481237;
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.094205617904663974) ) ) {
                    if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.07076432727538326;
                    } else {
                      result[0] += 0.07985107415049868;
                    }
                  } else {
                    result[0] += -0.07395888347949679;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.03824569748135415;
              } else {
                result[0] += -0.017324633910776283;
              }
            }
          } else {
            if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.07642362752062513;
            } else {
              result[0] += -0.004420102986848511;
            }
          }
        } else {
          if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += 0.01065829338328294;
          } else {
            result[0] += 0.05603529233366933;
          }
        }
      } else {
        if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.970608234405518466) ) ) {
          result[0] += 0.0439379554122514;
        } else {
          result[0] += -0.08892524545733696;
        }
      }
    } else {
      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.516936540603638583) ) ) {
          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
            if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                result[0] += 0.06405672985177106;
              } else {
                result[0] += -0.0946019192209339;
              }
            } else {
              result[0] += -0.08614877676694054;
            }
          } else {
            if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += -0.08453032426333854;
              } else {
                result[0] += 0.06124109228894545;
              }
            } else {
              result[0] += -0.09542248926136288;
            }
          }
        } else {
          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.516936540603638583) ) ) {
              result[0] += -0.09466237968376454;
            } else {
              if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.49770236015319913) ) ) {
                    result[0] += 0.024959855803387013;
                  } else {
                    result[0] += -0.059179780563160915;
                  }
                } else {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)120.0000000000000142) ) ) {
                    result[0] += 0.04370806720408593;
                  } else {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
                      result[0] += 0.16367135142611064;
                    } else {
                      result[0] += -0.08379341968203254;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                  result[0] += 0.055732877241464175;
                } else {
                  result[0] += -0.05880509789681165;
                }
              }
            }
          } else {
            result[0] += 0.03795967377820428;
          }
        }
      } else {
        if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
            if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)9.500000000000001776) ) ) {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.780892848968506748) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.13987779617309748) ) ) {
                    result[0] += 0.1421662456139031;
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
                      result[0] += -0.049781464007893635;
                    } else {
                      if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                        result[0] += 0.020222263291599447;
                      } else {
                        result[0] += 0.23499662664204976;
                      }
                    }
                  }
                } else {
                  result[0] += 0.1357014323266478;
                }
              } else {
                result[0] += -0.009637998473097023;
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                result[0] += -0.0386004872708784;
              } else {
                result[0] += 0.049719047649302586;
              }
            }
          } else {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
              if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                result[0] += -0.04690888064404865;
              } else {
                result[0] += -0.016269002563683053;
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.484580039978028232) ) ) {
                result[0] += 0.018839598743808986;
              } else {
                result[0] += -0.04887249855918098;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.894675970077515537) ) ) {
            result[0] += -0.0664911434595165;
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
                result[0] += -0.083219644870135;
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.605039834976196733) ) ) {
                  result[0] += -0.06673996478234644;
                } else {
                  result[0] += -0.0006195114420678085;
                }
              }
            } else {
              result[0] += 0.007169817971134198;
            }
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)12.50000000000000178) ) ) {
      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
        if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
            if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)11.50000000000000178) ) ) {
              result[0] += 0.04498079677603474;
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.232423543930054599) ) ) {
                result[0] += -0.07857605075113429;
              } else {
                result[0] += 0.021670041727039645;
              }
            }
          } else {
            if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.034426453123623316;
            } else {
              result[0] += -0.04111468250440949;
            }
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.94957673549652144) ) ) {
            result[0] += -0.03981705946838085;
          } else {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.261864185333252841) ) ) {
                result[0] += -0.0661551609043663;
              } else {
                if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)25.50000000000000355) ) ) {
                  result[0] += -0.02295967254957423;
                } else {
                  result[0] += 0.21070983480314742;
                }
              }
            } else {
              if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)31.50000000000000355) ) ) {
                result[0] += 0.006280724253567842;
              } else {
                result[0] += 0.049902796768607025;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
          result[0] += -0.04460100538758549;
        } else {
          result[0] += 0.031234181211679365;
        }
      }
    } else {
      result[0] += -0.06085163746848676;
    }
  }
  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)20.00000000000000355) ) ) {
    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.39281225204467951) ) ) {
        result[0] += -0.0008722011339510601;
      } else {
        result[0] += 0.016114103681334906;
      }
    } else {
      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.117121219635010654) ) ) {
            result[0] += -0.08619680746481796;
          } else {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
              result[0] += 0.1319045939325378;
            } else {
              result[0] += -0.06904920865223083;
            }
          }
        } else {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.516936540603638583) ) ) {
            if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.827801465988160068) ) ) {
              result[0] += -0.06488344335132198;
            } else {
              result[0] += 0.3333240865985795;
            }
          } else {
            if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
              result[0] += 0.007470056989753892;
            } else {
              if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.10697402110677652;
              } else {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.924581527709961826) ) ) {
                  result[0] += 0.2519717105769665;
                } else {
                  result[0] += 1.4679537186137857;
                }
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)10.50000000000000178) ) ) {
          if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)7.500000000000000888) ) ) {
            if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)6.500000000000000888) ) ) {
              if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)4.500000000000000888) ) ) {
                if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.004080637562204592;
                } else {
                  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
                    result[0] += -0.04931420777099417;
                  } else {
                    result[0] += 0.07257596044903018;
                  }
                }
              } else {
                result[0] += -0.039851959612259215;
              }
            } else {
              if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)11.50000000000000178) ) ) {
                result[0] += 0.006982227483797929;
              } else {
                result[0] += -0.056166439448506106;
              }
            }
          } else {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.599987030029298651) ) ) {
                result[0] += 0.1170225374965094;
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.206118345260621005) ) ) {
                  result[0] += -0.07608536249840937;
                } else {
                  result[0] += 0.08475313305577196;
                }
              }
            } else {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.516936540603638583) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.1822080612182635) ) ) {
                    result[0] += 0.0741674797940859;
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.778982400894165927) ) ) {
                      result[0] += -0.08522797765941136;
                    } else {
                      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
                        result[0] += -0.07485526879088064;
                      } else {
                        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                          result[0] += -0.013459089384035786;
                        } else {
                          result[0] += 0.1679051226257764;
                        }
                      }
                    }
                  }
                } else {
                  result[0] += 0.048520920429879086;
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.3070559501647967) ) ) {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.516936540603638583) ) ) {
                      result[0] += 0.0021713580744405405;
                    } else {
                      result[0] += -0.08444653480527176;
                    }
                  } else {
                    result[0] += -0.11300395518711358;
                  }
                } else {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.02972401232962449;
                    } else {
                      result[0] += 0.02220897130307722;
                    }
                  } else {
                    if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                      result[0] += 0.016605537004742685;
                    } else {
                      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.172047138214112216) ) ) {
                          result[0] += 0.07171546181553543;
                        } else {
                          result[0] += -0.05600980504797936;
                        }
                      } else {
                        result[0] += -0.1271123071275355;
                      }
                    }
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.694163918495178667) ) ) {
            if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)6.500000000000000888) ) ) {
              result[0] += 0.04733618866156441;
            } else {
              result[0] += -0.04410011343454892;
            }
          } else {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
              if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)31.50000000000000355) ) ) {
                result[0] += -0.007420053109641105;
              } else {
                result[0] += 0.04788747815579894;
              }
            } else {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                result[0] += -0.06090302671041908;
              } else {
                result[0] += 0.044771893234541504;
              }
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)40.00000000000000711) ) ) {
      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.827801465988160068) ) ) {
        result[0] += -0.019509440948171676;
      } else {
        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)56.00000000000000711) ) ) {
          result[0] += -0.09278741385912082;
        } else {
          result[0] += -0.016086246966794306;
        }
      }
    } else {
      if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)7.000000000000000888) ) ) {
        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.909855604171753818) ) ) {
            if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.888826131820679155) ) ) {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += 0.04124103246323426;
                    } else {
                      result[0] += -0.013155264614618767;
                    }
                  } else {
                    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += -0.07192806663443635;
                    } else {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.7512402534484881) ) ) {
                        result[0] += 0.010257772538876919;
                      } else {
                        if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.935600519180298074) ) ) {
                          result[0] += -0.03846912748962871;
                        } else {
                          result[0] += 0.051405013322349695;
                        }
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.978769779205324042) ) ) {
                    result[0] += -0.06376164491979096;
                  } else {
                    result[0] += 0.032875457879233426;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                  result[0] += -0.007835046571728347;
                } else {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.058688038657493814;
                  } else {
                    result[0] += 0.032796197642174674;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                result[0] += 0.01635382391087332;
              } else {
                result[0] += -0.01859047018933747;
              }
            }
          } else {
            result[0] += -0.028330529309135696;
          }
        } else {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.464467763900757724) ) ) {
            result[0] += -0.030677810147444508;
          } else {
            result[0] += 0.07393661944499788;
          }
        }
      } else {
        if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.484580039978028232) ) ) {
            result[0] += 0.006486516096664556;
          } else {
            result[0] += -0.02598995923444987;
          }
        } else {
          result[0] += -0.09952117020150497;
        }
      }
    }
  }
  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)20.00000000000000355) ) ) {
    if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)10.50000000000000178) ) ) {
      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)3.000000000000000444) ) ) {
          if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
            result[0] += 0.006053395072105939;
          } else {
            result[0] += 0.03055265368694028;
          }
        } else {
          if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
              result[0] += 0.09689877261729946;
            } else {
              if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += -0.06968297829063576;
              } else {
                result[0] += -0.015050078364737849;
              }
            }
          } else {
            if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += -0.029951506814393104;
            } else {
              if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.232423543930054599) ) ) {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.172047138214112216) ) ) {
                        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                          result[0] += 0.0571486036463453;
                        } else {
                          if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)1.500000000000000222) ) ) {
                            result[0] += -0.005094072627655689;
                          } else {
                            result[0] += -0.05281972028491684;
                          }
                        }
                      } else {
                        result[0] += 0.05075451864193038;
                      }
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.39281225204467951) ) ) {
                        result[0] += 0.0032455819670705165;
                      } else {
                        result[0] += 0.07367010511489244;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.06448161497817656;
                    } else {
                      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.481121778488159624) ) ) {
                          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                            if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                              result[0] += -0.06114995899044941;
                            } else {
                              result[0] += 0.014748510911427135;
                            }
                          } else {
                            result[0] += 0.040225480894369;
                          }
                        } else {
                          result[0] += -0.030575142436532927;
                        }
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.302512168884278232) ) ) {
                          result[0] += 0.01650553060889817;
                        } else {
                          result[0] += -0.047301146759959306;
                        }
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.484580039978028232) ) ) {
                    result[0] += -0.06821379471104437;
                  } else {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.802901029586792436) ) ) {
                      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                          result[0] += -0.06995475055536522;
                        } else {
                          result[0] += 0.041384956518377196;
                        }
                      } else {
                        result[0] += -0.08462003249672631;
                      }
                    } else {
                      result[0] += 0.031921236104805246;
                    }
                  }
                }
              } else {
                result[0] += 0.03723233196104858;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)12.50000000000000178) ) ) {
          if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)6.500000000000000888) ) ) {
            if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)4.500000000000000888) ) ) {
              result[0] += -0.005147272660150661;
            } else {
              result[0] += -0.04052067990219352;
            }
          } else {
            if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)11.50000000000000178) ) ) {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.623839378356934482) ) ) {
                    result[0] += 0.05857827719089145;
                  } else {
                    result[0] += -0.004262004206771351;
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += -0.09343490074330563;
                    } else {
                      result[0] += -0.002512138005148012;
                    }
                  } else {
                    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.07018072131554205;
                    } else {
                      result[0] += 0.013290283514498881;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.651049375534058505) ) ) {
                  if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.027194624777685694;
                  } else {
                    if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)3.500000000000000444) ) ) {
                      result[0] += 0.12790012875633502;
                    } else {
                      result[0] += 0.019341147113602873;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.802696108818054643) ) ) {
                    result[0] += 0.0327164997648075;
                  } else {
                    result[0] += -0.06458235975688349;
                  }
                }
              }
            } else {
              result[0] += -0.06490367971857951;
            }
          }
        } else {
          result[0] += -0.07067471502883495;
        }
      }
    } else {
      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.802696108818054643) ) ) {
        if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)6.500000000000000888) ) ) {
          if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
            result[0] += -0.020216628601530267;
          } else {
            result[0] += 0.09160742235025283;
          }
        } else {
          result[0] += -0.04390148571175405;
        }
      } else {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
          result[0] += 0.010928051787869257;
        } else {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
            result[0] += -0.057933623224572975;
          } else {
            result[0] += 0.04234168685858068;
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)80.00000000000001421) ) ) {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.834949493408204901) ) ) {
        result[0] += 0.019119785350972467;
      } else {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
          if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.515218973159790483) ) ) {
              result[0] += 0.05570509779275184;
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.039720773696899636) ) ) {
                  result[0] += -0.004326271444491735;
                } else {
                  result[0] += 0.06809289223264813;
                }
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.484580039978028232) ) ) {
                  result[0] += -0.008714853056829768;
                } else {
                  result[0] += -0.07985821688437934;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                  result[0] += 0.01720376077929565;
                } else {
                  result[0] += -0.05330871594958802;
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.1822080612182635) ) ) {
                  result[0] += 0.12795727482674996;
                } else {
                  result[0] += 0.003542542450133366;
                }
              }
            } else {
              if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += -0.08390789857780906;
              } else {
                result[0] += -0.03308419672321929;
              }
            }
          }
        } else {
          result[0] += -0.0064196651815691015;
        }
      }
    } else {
      if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.445705175399781162) ) ) {
          result[0] += 0.008780479005736882;
        } else {
          result[0] += -0.008800327612882436;
        }
      } else {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.909855604171753818) ) ) {
          result[0] += -0.03461864759500592;
        } else {
          result[0] += 0.025217584781504238;
        }
      }
    }
  }
  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)20.00000000000000355) ) ) {
    if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
      if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.1822080612182635) ) ) {
          if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)3.500000000000000444) ) ) {
            result[0] += 0.0006286774545151431;
          } else {
            result[0] += -0.07516624797104055;
          }
        } else {
          result[0] += 0.012476313639404582;
        }
      } else {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
          if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.938867926597595659) ) ) {
            result[0] += 0.01907475864247758;
          } else {
            result[0] += -0.0768057566348166;
          }
        } else {
          if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)31.50000000000000355) ) ) {
            if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)7.500000000000000888) ) ) {
              if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.827801465988160068) ) ) {
                    result[0] += -0.0032655908768924515;
                  } else {
                    result[0] += -0.06044126999180355;
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.744781017303467685) ) ) {
                    result[0] += 0.023272043962682693;
                  } else {
                    result[0] += -0.026690960063154825;
                  }
                }
              } else {
                if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
                  result[0] += -0.048073420714290105;
                } else {
                  result[0] += 0.05181768834067566;
                }
              }
            } else {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.599987030029298651) ) ) {
                  result[0] += 0.0934320182070702;
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.206118345260621005) ) ) {
                    result[0] += -0.0783494261247151;
                  } else {
                    result[0] += 0.050408582828871944;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.516936540603638583) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.1822080612182635) ) ) {
                      result[0] += 0.06977550830844047;
                    } else {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.778982400894165927) ) ) {
                        result[0] += -0.0830707196538622;
                      } else {
                        if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                          result[0] += -0.06199358893725755;
                        } else {
                          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                            result[0] += -0.009713918360143362;
                          } else {
                            result[0] += 0.141625549869286;
                          }
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.516936540603638583) ) ) {
                        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.516936540603638583) ) ) {
                          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)120.0000000000000142) ) ) {
                            if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                              result[0] += 0.06809254021144653;
                            } else {
                              result[0] += -0.014481575375982852;
                            }
                          } else {
                            result[0] += -0.07555495353887265;
                          }
                        } else {
                          result[0] += -0.07210003511447871;
                        }
                      } else {
                        result[0] += 0.05544660343988921;
                      }
                    } else {
                      if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += -0.04796133214303852;
                      } else {
                        result[0] += 0.0747989103942793;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.3070559501647967) ) ) {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.040716171264650214) ) ) {
                        result[0] += -0.13112507154739755;
                      } else {
                        result[0] += -0.009992509875410895;
                      }
                    } else {
                      result[0] += -0.10252855568696273;
                    }
                  } else {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += -0.027492244894895098;
                      } else {
                        result[0] += 0.02040052161378015;
                      }
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.172047138214112216) ) ) {
                        result[0] += 0.02355357476687067;
                      } else {
                        if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                          result[0] += -0.009790661501192153;
                        } else {
                          result[0] += -0.08785946415565732;
                        }
                      }
                    }
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.255632162094117099) ) ) {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += -0.11989461919214671;
              } else {
                result[0] += 0.013277177082934855;
              }
            } else {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                result[0] += 0.10236099470731826;
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.161602735519410068) ) ) {
                  result[0] += 0.05897968624504665;
                } else {
                  if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)32.50000000000000711) ) ) {
                    result[0] += -0.1791804935894282;
                  } else {
                    result[0] += 0.011486473196654183;
                  }
                }
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.802696108818054643) ) ) {
        result[0] += -0.08176831026989913;
      } else {
        if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)2.500000000000000444) ) ) {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.208071470260621005) ) ) {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.057478274291857645;
              } else {
                result[0] += -0.058662069671493136;
              }
            } else {
              result[0] += 0.058826432602408246;
            }
          } else {
            result[0] += 0.07253901381079322;
          }
        } else {
          if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)11.50000000000000178) ) ) {
            result[0] += 0.05940258050666321;
          } else {
            result[0] += 0.007971754980234936;
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.888826131820679155) ) ) {
        result[0] += -0.016009614009224984;
      } else {
        result[0] += -0.09565667115407514;
      }
    } else {
      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.736135363578796831) ) ) {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
          if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += 0.163658345431002;
          } else {
            result[0] += -0.0029078478757007383;
          }
        } else {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
            result[0] += -0.05292831530689407;
          } else {
            result[0] += 0.030101126847883405;
          }
        }
      } else {
        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
          if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.302512168884278232) ) ) {
              result[0] += -0.009836276031142566;
            } else {
              result[0] += -0.04650019450824557;
            }
          } else {
            if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)7.000000000000000888) ) ) {
              if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)7.000000000000000888) ) ) {
                result[0] += 0.008114093190653973;
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.909855604171753818) ) ) {
                  result[0] += -0.029699538699132902;
                } else {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += 0.045047421845850866;
                  } else {
                    result[0] += -0.055783343562947965;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.718933820724488193) ) ) {
                  result[0] += -0.005553289839079805;
                } else {
                  result[0] += -0.05060989036992663;
                }
              } else {
                result[0] += -0.101957095450788;
              }
            }
          }
        } else {
          result[0] += -0.03616051925852023;
        }
      }
    }
  }
  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)20.00000000000000355) ) ) {
    if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)10.50000000000000178) ) ) {
      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
          if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)3.000000000000000444) ) ) {
            if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)8.500000000000001776) ) ) {
                result[0] += 0.006845887748401818;
              } else {
                result[0] += -0.05708924499640318;
              }
            } else {
              result[0] += 0.0283415433304096;
            }
          } else {
            if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)7.500000000000000888) ) ) {
              if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.005555029586758813;
              } else {
                result[0] += -0.07006690254338324;
              }
            } else {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += -0.02740023181185532;
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.736135363578796831) ) ) {
                    result[0] += -0.11886849440540759;
                  } else {
                    result[0] += 0.005988249745555801;
                  }
                }
              } else {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.302512168884278232) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.1822080612182635) ) ) {
                      result[0] += 0.06427500559709824;
                    } else {
                      if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += -0.06862521747057629;
                      } else {
                        if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                          result[0] += -0.08616819771246396;
                        } else {
                          result[0] += 0.13335429451357098;
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.172047138214112216) ) ) {
                      if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                        result[0] += -0.08035538144004525;
                      } else {
                        result[0] += 0.05450350806629162;
                      }
                    } else {
                      if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += 0.011825953098463538;
                      } else {
                        result[0] += 0.05340744783209064;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.842307567596437323) ) ) {
                    result[0] += -0.06972776521430084;
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
                      if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                        result[0] += 0.02547575710872308;
                      } else {
                        result[0] += -0.07804657026391366;
                      }
                    } else {
                      if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                        if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += -0.04881295620821205;
                        } else {
                          if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)1.500000000000000222) ) ) {
                            result[0] += -0.040435045318401724;
                          } else {
                            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.142630577087403232) ) ) {
                              result[0] += -0.06823502916135267;
                            } else {
                              result[0] += 0.02734374290946758;
                            }
                          }
                        }
                      } else {
                        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                          result[0] += -0.023722239547786284;
                        } else {
                          result[0] += -0.10204991839323828;
                        }
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
              result[0] += 0.033098760717338915;
            } else {
              result[0] += -0.04334386805422759;
            }
          } else {
            result[0] += 0.07710980891778203;
          }
        }
      } else {
        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.651049375534058505) ) ) {
            if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)13.50000000000000178) ) ) {
              result[0] += -0.029877143199258355;
            } else {
              if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)15.50000000000000178) ) ) {
                result[0] += 0.0574830297002408;
              } else {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                  result[0] += 0.05412795265132039;
                } else {
                  result[0] += -0.04839953294678984;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.05582100544252094;
            } else {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.039720773696899636) ) ) {
                result[0] += 0.00745576378651852;
              } else {
                result[0] += -0.5986134096028467;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += 0.01741988010389693;
          } else {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.0730634862971434;
            } else {
              result[0] += -0.002344792056523116;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.802696108818054643) ) ) {
        if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)6.500000000000000888) ) ) {
          if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
            result[0] += -0.019339574368777215;
          } else {
            result[0] += 0.08581510986571841;
          }
        } else {
          result[0] += -0.04000036010229133;
        }
      } else {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
          result[0] += 0.011217946765447599;
        } else {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
            result[0] += -0.05521783255084767;
          } else {
            result[0] += 0.03808470036576484;
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)80.00000000000001421) ) ) {
      if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.801661729812622958) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.205872535705568183) ) ) {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += 0.021919274501915657;
            } else {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                result[0] += -0.18044573999476848;
              } else {
                result[0] += 0.021283665549021717;
              }
            }
          } else {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.008786460022535466;
              } else {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.44140100479126021) ) ) {
                    result[0] += -0.08817062836806887;
                  } else {
                    result[0] += 0.13548523290104106;
                  }
                } else {
                  if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.0513756687703865;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.97661352157592951) ) ) {
                      result[0] += 0.04790458901040489;
                    } else {
                      result[0] += -0.026140817275766864;
                    }
                  }
                }
              }
            } else {
              result[0] += -0.006158212423096603;
            }
          }
        } else {
          result[0] += -0.06728872271120102;
        }
      } else {
        result[0] += 0.0258297163878993;
      }
    } else {
      if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.605039834976196733) ) ) {
          result[0] += -0.10243185740876602;
        } else {
          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
              result[0] += -0.1416954407189451;
            } else {
              result[0] += -0.030746974256898343;
            }
          } else {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += 0.004144652814392681;
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.90474271774292081) ) ) {
                result[0] += -0.06885394121543824;
              } else {
                result[0] += 0.03017259980385489;
              }
            }
          }
        }
      } else {
        result[0] += 0.003209420996301618;
      }
    }
  }
  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)20.00000000000000355) ) ) {
    if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
      if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
        if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)11.50000000000000178) ) ) {
          if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)257681260544.0000305) ) ) {
              if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                result[0] += 0.06717664894468602;
              } else {
                result[0] += 0.00314396487944954;
              }
            } else {
              result[0] += 0.033214596822367624;
            }
          } else {
            if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)6.500000000000000888) ) ) {
              if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)7.500000000000000888) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.284418344497681552) ) ) {
                  if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)2.500000000000000444) ) ) {
                    if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.006240698046097796;
                    } else {
                      result[0] += -0.11626467679135676;
                    }
                  } else {
                    if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)4.500000000000000888) ) ) {
                      result[0] += 0.05837190028670937;
                    } else {
                      result[0] += -0.017948981000501896;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)2.500000000000000444) ) ) {
                    if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2415.000000000000455) ) ) {
                      result[0] += -0.016371315441649503;
                    } else {
                      result[0] += 0.056403451372161385;
                    }
                  } else {
                    if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                      result[0] += -0.0692880968920833;
                    } else {
                      result[0] += -0.020355838322291625;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += -0.02683384579114454;
                  } else {
                    result[0] += 0.0035579631836770327;
                  }
                } else {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.142630577087403232) ) ) {
                      result[0] += -0.08777155696419123;
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.516936540603638583) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.1822080612182635) ) ) {
                          result[0] += 0.06060460545490848;
                        } else {
                          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                            result[0] += -0.09233648446747646;
                          } else {
                            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                              result[0] += -0.02059068537391169;
                            } else {
                              result[0] += 0.0846911702295225;
                            }
                          }
                        }
                      } else {
                        result[0] += 0.03767492062013393;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.3070559501647967) ) ) {
                      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                        result[0] += -0.018710387394012633;
                      } else {
                        result[0] += -0.0954924061009963;
                      }
                    } else {
                      if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += 0.012823363408560877;
                      } else {
                        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)120.0000000000000142) ) ) {
                          result[0] += 0.00731738965838873;
                        } else {
                          result[0] += -0.06271354508691505;
                        }
                      }
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.825422286987305576) ) ) {
                if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.0022261730052104715;
                } else {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.433569431304932529) ) ) {
                    result[0] += 0.028646439798695623;
                  } else {
                    if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)29.50000000000000355) ) ) {
                      result[0] += -0.04605522878181965;
                    } else {
                      result[0] += 0.03682272103977301;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)7.500000000000000888) ) ) {
                  if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)17.50000000000000355) ) ) {
                    result[0] += 0.014047337636466454;
                  } else {
                    result[0] += -0.09030475576606933;
                  }
                } else {
                  result[0] += -0.08775305975351505;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += -0.05853169166854009;
          } else {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.607751369476319248) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.607751369476319248) ) ) {
                result[0] += -0.11058436903503524;
              } else {
                result[0] += 0.035061666690416556;
              }
            } else {
              result[0] += 0.037874792241090444;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.357691764831543413) ) ) {
          result[0] += -0.06852705264882235;
        } else {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.03776322404534725;
            } else {
              result[0] += -0.03854339532390366;
            }
          } else {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.42478513717651456) ) ) {
              result[0] += 0.1059347733900742;
            } else {
              result[0] += -0.012087138423812135;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.827801465988160068) ) ) {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.172047138214112216) ) ) {
          result[0] += -0.07713800331183038;
        } else {
          result[0] += 0.0011340732205214821;
        }
      } else {
        if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)2.500000000000000444) ) ) {
          result[0] += 0.04961354633821584;
        } else {
          result[0] += 0.014204299987307871;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.736135363578796831) ) ) {
        result[0] += -0.006824889842494375;
      } else {
        if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.500000000000000444) ) ) {
          result[0] += -0.061332101482174085;
        } else {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
            result[0] += 0.01846990917854439;
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.87548160552978693) ) ) {
              result[0] += -0.04502303436463239;
            } else {
              result[0] += 0.05351744189204242;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)2.500000000000000444) ) ) {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.58381557464599787) ) ) {
            result[0] += -0.005968999412616848;
          } else {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.10742906199399954;
            } else {
              result[0] += -0.06917605526106692;
            }
          }
        } else {
          result[0] += 0.026810296492646588;
        }
      } else {
        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
          if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.25930547714233576) ) ) {
              if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)7.000000000000000888) ) ) {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += 0.01211940096904168;
                } else {
                  if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += 0.016287135787302842;
                  } else {
                    result[0] += -0.086791673471112;
                  }
                }
              } else {
                result[0] += -0.05082537909270473;
              }
            } else {
              if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)3.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.827801465988160068) ) ) {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
                    result[0] += -0.06797271307654017;
                  } else {
                    result[0] += 0.01761848869826475;
                  }
                } else {
                  result[0] += -0.08790554520582733;
                }
              } else {
                result[0] += -0.007744996570764603;
              }
            }
          } else {
            result[0] += 0.042777942431602746;
          }
        } else {
          result[0] += -0.031876393592559414;
        }
      }
    }
  }
  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
      if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)11.50000000000000178) ) ) {
        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)20.00000000000000355) ) ) {
          if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)120.0000000000000142) ) ) {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.909855604171753818) ) ) {
                if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)5.500000000000000888) ) ) {
                    if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)7.500000000000000888) ) ) {
                      if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += 0.008695804963314378;
                        } else {
                          result[0] += -0.007376256201757136;
                        }
                      } else {
                        result[0] += -0.039527980344337396;
                      }
                    } else {
                      if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                        result[0] += 0.013442031695938206;
                      } else {
                        result[0] += -0.06644098565541005;
                      }
                    }
                  } else {
                    result[0] += -0.08703177312979976;
                  }
                } else {
                  if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)3.500000000000000444) ) ) {
                    if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                      result[0] += 0.44859515918448123;
                    } else {
                      result[0] += 0.10163457595620347;
                    }
                  } else {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.605039834976196733) ) ) {
                      if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)8.500000000000001776) ) ) {
                        if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)6.500000000000000888) ) ) {
                          result[0] += 0.051014944627215245;
                        } else {
                          if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                            result[0] += -0.05586400291235921;
                          } else {
                            result[0] += 0.06299966537461978;
                          }
                        }
                      } else {
                        if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)30.50000000000000355) ) ) {
                          result[0] += 0.07212974021981644;
                        } else {
                          result[0] += -0.03012039290588477;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.605039834976196733) ) ) {
                          result[0] += 0.0020440846425894372;
                        } else {
                          result[0] += -0.06691074221627595;
                        }
                      } else {
                        if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)9.500000000000001776) ) ) {
                          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.849175214767456943) ) ) {
                            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                              if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                                result[0] += 0.12634337139070415;
                              } else {
                                result[0] += 0.02808350545900832;
                              }
                            } else {
                              if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
                                result[0] += -0.04263296601904731;
                              } else {
                                result[0] += 0.06460909667248364;
                              }
                            }
                          } else {
                            result[0] += -0.03544584817379602;
                          }
                        } else {
                          result[0] += -0.005612338708389867;
                        }
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)9.500000000000001776) ) ) {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                    if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)14.50000000000000178) ) ) {
                      if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                        if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
                            if ( UNLIKELY(  (data[27].missing != -1) && (data[27].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                              result[0] += -0.2661069639876749;
                            } else {
                              result[0] += 0.0018809651519997726;
                            }
                          } else {
                            result[0] += -0.019136379641412866;
                          }
                        } else {
                          if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
                            result[0] += 0.11797345970634232;
                          } else {
                            result[0] += -0.03074270757895613;
                          }
                        }
                      } else {
                        if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)4.500000000000000888) ) ) {
                          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                            result[0] += 0.013761411726688454;
                          } else {
                            if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
                              result[0] += -0.007963906096442283;
                            } else {
                              result[0] += 0.10744080212422345;
                            }
                          }
                        } else {
                          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.718933820724488193) ) ) {
                            if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)21.50000000000000355) ) ) {
                              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.651049375534058505) ) ) {
                                result[0] += 0.14563901944749957;
                              } else {
                                result[0] += -0.10200554552209518;
                              }
                            } else {
                              result[0] += 0.040825778875130316;
                            }
                          } else {
                            result[0] += -0.10511337963544615;
                          }
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.827801465988160068) ) ) {
                        result[0] += -0.17602246911101682;
                      } else {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.644374847412110263) ) ) {
                          result[0] += 0.09774897900208453;
                        } else {
                          result[0] += -0.12430723489220466;
                        }
                      }
                    }
                  } else {
                    result[0] += -0.019391420623083657;
                  }
                } else {
                  result[0] += -0.058184077557174176;
                }
              }
            } else {
              if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                  result[0] += -0.004813491081813891;
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
                    result[0] += -0.051767491819617865;
                  } else {
                    result[0] += 0.09526875488640049;
                  }
                }
              } else {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
                  result[0] += -0.0616278514116005;
                } else {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                    if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += 0.023892475730247414;
                    } else {
                      result[0] += -0.05755193967607678;
                    }
                  } else {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.029068946838379794) ) ) {
                      if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                        result[0] += -0.04915265584480304;
                      } else {
                        result[0] += -0.13967208627125693;
                      }
                    } else {
                      if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                        result[0] += -0.066191412055488;
                      } else {
                        result[0] += 0.06921035495560098;
                      }
                    }
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.484580039978028232) ) ) {
              result[0] += -0.10216610163297707;
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                result[0] += 0.015993888111473578;
              } else {
                result[0] += 0.07555889239070686;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)80.00000000000001421) ) ) {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.801661729812622958) ) ) {
              result[0] += -0.01521236914975169;
            } else {
              result[0] += -0.06370135129057909;
            }
          } else {
            result[0] += -0.0015252641288956053;
          }
        }
      } else {
        result[0] += -0.057362826466840655;
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.02604460716247603) ) ) {
        result[0] += -0.09813018295313271;
      } else {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += 0.03811208158690635;
          } else {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += -0.07051820937983914;
            } else {
              result[0] += 0.03213362854918511;
            }
          }
        } else {
          result[0] += 0.09582326840858471;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.802696108818054643) ) ) {
      result[0] += -0.08585183230290142;
    } else {
      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.938867926597595659) ) ) {
        result[0] += -0.08213721656764078;
      } else {
        result[0] += 0.023511455510118528;
      }
    }
  }
  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)20.00000000000000355) ) ) {
    if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
      if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
            result[0] += 0.06881533618464698;
          } else {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.42478513717651456) ) ) {
              result[0] += 0.0007704223708575964;
            } else {
              result[0] += 0.036308091329926705;
            }
          }
        } else {
          result[0] += 0.033675651571573116;
        }
      } else {
        if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
          if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)12.50000000000000178) ) ) {
            if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)5.500000000000000888) ) ) {
              if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)5.500000000000000888) ) ) {
                if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)10.50000000000000178) ) ) {
                      if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)9.500000000000001776) ) ) {
                        if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)8.500000000000001776) ) ) {
                          result[0] += -0.0013317314582053773;
                        } else {
                          if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                            result[0] += -0.10730162948935595;
                          } else {
                            result[0] += 0.03658162414781253;
                          }
                        }
                      } else {
                        result[0] += 0.026635945533334385;
                      }
                    } else {
                      result[0] += -0.04744629060700102;
                    }
                  } else {
                    result[0] += 0.09734728543257581;
                  }
                } else {
                  result[0] += -0.03825596937258571;
                }
              } else {
                result[0] += -0.06675990136723574;
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.094205617904663974) ) ) {
                if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)13.50000000000000178) ) ) {
                  if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)11.50000000000000178) ) ) {
                    if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)9.500000000000001776) ) ) {
                      result[0] += -0.060372107152055654;
                    } else {
                      result[0] += 0.04033434712852021;
                    }
                  } else {
                    result[0] += -0.10045337730394562;
                  }
                } else {
                  result[0] += 0.04420676846489554;
                }
              } else {
                if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.605039834976196733) ) ) {
                    result[0] += -0.000856824016094907;
                  } else {
                    result[0] += -0.08147254355194218;
                  }
                } else {
                  if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.500000000000000444) ) ) {
                    if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
                      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                        result[0] += 0.02448637752152991;
                      } else {
                        result[0] += -0.1232674426951726;
                      }
                    } else {
                      result[0] += 0.08573748756959167;
                    }
                  } else {
                    if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)13.50000000000000178) ) ) {
                      result[0] += 0.013037235264086492;
                    } else {
                      if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += 0.017256948687447555;
                      } else {
                        if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)26.50000000000000355) ) ) {
                          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.827801465988160068) ) ) {
                            result[0] += -0.00357813998240554;
                          } else {
                            result[0] += -0.08050703436566164;
                          }
                        } else {
                          result[0] += -0.008322636299940544;
                        }
                      }
                    }
                  }
                }
              }
            }
          } else {
            result[0] += -0.0634608324869622;
          }
        } else {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)120.0000000000000142) ) ) {
            if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.03349975514874897;
            } else {
              result[0] += 0.046290622747081414;
            }
          } else {
            result[0] += -0.021904331428853292;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.02604460716247603) ) ) {
        result[0] += -0.11976256100752379;
      } else {
        if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)2.500000000000000444) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.208071470260621005) ) ) {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.05105951958144239;
              } else {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.10185561070707363;
                } else {
                  if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += 0.06522613342476469;
                  } else {
                    result[0] += -0.039777310891457855;
                  }
                }
              }
            } else {
              result[0] += 0.07148605304854017;
            }
          } else {
            result[0] += 0.06231212372921962;
          }
        } else {
          if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)16.50000000000000355) ) ) {
            result[0] += 0.012912974266846589;
          } else {
            result[0] += -0.057206259950474896;
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)40.00000000000000711) ) ) {
      if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.515218973159790483) ) ) {
          result[0] += 0.05837315462099059;
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.500000000000000444) ) ) {
              result[0] += -0.06297929818006313;
            } else {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.039720773696899636) ) ) {
                result[0] += 0.011012753978875663;
              } else {
                result[0] += 0.0813746588528867;
              }
            }
          } else {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.172047138214112216) ) ) {
              result[0] += -0.011865728605439434;
            } else {
              result[0] += -0.09602400811017661;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.497866153717041238) ) ) {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.987438440322876421) ) ) {
                result[0] += 0.10121186690489503;
              } else {
                if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                  result[0] += 0.14191292120860738;
                } else {
                  result[0] += -0.06686880394366695;
                }
              }
            } else {
              result[0] += -0.06970366243458302;
            }
          } else {
            result[0] += -0.06290259851987921;
          }
        } else {
          result[0] += -0.007844664559481887;
        }
      }
    } else {
      if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)7.000000000000000888) ) ) {
        if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.736135363578796831) ) ) {
            result[0] += 0.0031288114152575163;
          } else {
            if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
              result[0] += -0.039637123989511906;
            } else {
              result[0] += -0.0041229333361542725;
            }
          }
        } else {
          if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
            result[0] += 0.022093345650686408;
          } else {
            result[0] += 0.002216621498728363;
          }
        }
      } else {
        if ( UNLIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += -0.0971473438694419;
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.607751369476319248) ) ) {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
              if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                  result[0] += -0.060988418513469735;
                } else {
                  result[0] += 0.0711147349500207;
                }
              } else {
                result[0] += -0.04822295083659123;
              }
            } else {
              result[0] += 0.018439183509277556;
            }
          } else {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.607751369476319248) ) ) {
              result[0] += -0.042766631699484935;
            } else {
              result[0] += 0.0018861684795100952;
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)20.00000000000000355) ) ) {
    if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.790835380554201) ) ) {
          if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
            if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.06873916731161415;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.516936540603638583) ) ) {
                  result[0] += -0.10799941899702221;
                } else {
                  result[0] += 0.004983770692075832;
                }
              }
            } else {
              if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.827801465988160068) ) ) {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.802901029586792436) ) ) {
                  result[0] += -0.08623253155410862;
                } else {
                  result[0] += -0.027422402545526744;
                }
              } else {
                result[0] += 0.05207770489823668;
              }
            }
          } else {
            result[0] += 0.0028540271086241113;
          }
        } else {
          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)8.500000000000001776) ) ) {
              result[0] += 0.015925909910163384;
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.232423543930054599) ) ) {
                if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)31.50000000000000355) ) ) {
                  result[0] += -0.05265356663659379;
                } else {
                  result[0] += -0.29479185514103556;
                }
              } else {
                if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)10.50000000000000178) ) ) {
                  result[0] += -0.0769527229140049;
                } else {
                  result[0] += 0.012363445977506413;
                }
              }
            }
          } else {
            result[0] += -0.0048128899856227855;
          }
        }
      } else {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.172047138214112216) ) ) {
            result[0] += -0.08477950711858262;
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.718933820724488193) ) ) {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.020894599006856456;
              } else {
                result[0] += -0.08715035552853248;
              }
            } else {
              if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)25.50000000000000355) ) ) {
                result[0] += 0.01118768779620039;
              } else {
                result[0] += 0.15018438751380625;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.894675970077515537) ) ) {
            result[0] += -0.05810715951688469;
          } else {
            if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)2.65906000137329146) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.95235633850097834) ) ) {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                      result[0] += 0.020337575146149534;
                    } else {
                      result[0] += -0.09396849865995782;
                    }
                  } else {
                    result[0] += -0.024446939907556045;
                  }
                } else {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
                      result[0] += -0.04297463166197388;
                    } else {
                      if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                        if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
                          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                            result[0] += 0.03579427656409536;
                          } else {
                            result[0] += -0.0024815990356623647;
                          }
                        } else {
                          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                            result[0] += -0.052541546883433446;
                          } else {
                            result[0] += -0.008012425325690296;
                          }
                        }
                      } else {
                        result[0] += 0.03083103369260061;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
                      result[0] += 0.04667450878728626;
                    } else {
                      if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                        result[0] += -0.07975092092604014;
                      } else {
                        result[0] += 0.0019006790123196574;
                      }
                    }
                  }
                }
              } else {
                result[0] += -0.1616195120044466;
              }
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.484580039978028232) ) ) {
                result[0] += -0.07191199405508397;
              } else {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.909855604171753818) ) ) {
                    result[0] += -0.01911204901730862;
                  } else {
                    result[0] += 0.06447042474728241;
                  }
                } else {
                  result[0] += 0.07237281599254274;
                }
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.284418344497681552) ) ) {
        result[0] += -0.024738580834967612;
      } else {
        if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)2.500000000000000444) ) ) {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)17.50000000000000355) ) ) {
              result[0] += -0.11467392202717269;
            } else {
              result[0] += 0.09465318370323167;
            }
          } else {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += -0.04550073985039655;
            } else {
              result[0] += 0.0518025285193964;
            }
          }
        } else {
          result[0] += 0.014044795328859983;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)40.00000000000000711) ) ) {
      if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
        result[0] += -0.0016862713537647732;
      } else {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
          if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.87254357337951749) ) ) {
              result[0] += 0.22604757390938102;
            } else {
              result[0] += -0.03318892751158297;
            }
          } else {
            result[0] += -0.09224959486244363;
          }
        } else {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.418141007423401323) ) ) {
            if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.2186163905102145;
            } else {
              result[0] += -0.004898866886801838;
            }
          } else {
            if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.08379279199334398;
            } else {
              result[0] += -0.019002477683109628;
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)7.000000000000000888) ) ) {
        if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.605039834976196733) ) ) {
            result[0] += -0.10676883954177992;
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.067782521247864214) ) ) {
              result[0] += -0.12111930725457559;
            } else {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                  result[0] += -0.12736554874052053;
                } else {
                  result[0] += -0.021060694888926767;
                }
              } else {
                if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                  result[0] += 0.027856856821254123;
                } else {
                  if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.802696108818054643) ) ) {
                    result[0] += -0.03043405274008549;
                  } else {
                    if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
                      result[0] += 0.007291883994803306;
                    } else {
                      result[0] += -0.08149136327165214;
                    }
                  }
                }
              }
            }
          }
        } else {
          result[0] += 0.0061255785959683584;
        }
      } else {
        if ( UNLIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += -0.09559469016555833;
        } else {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.718933820724488193) ) ) {
            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
              result[0] += -0.010941558218800891;
            } else {
              result[0] += 0.03675947477005811;
            }
          } else {
            result[0] += -0.05107592918781765;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)17.50000000000000355) ) ) {
    if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)20.00000000000000355) ) ) {
      if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
        if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
          if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
              result[0] += 0.0613330560380026;
            } else {
              result[0] += 0.0032880637487997023;
            }
          } else {
            result[0] += 0.035399807230599385;
          }
        } else {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.795762062072754794) ) ) {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)120.0000000000000142) ) ) {
              if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)14.50000000000000178) ) ) {
                if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)13.50000000000000178) ) ) {
                  if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)7.500000000000000888) ) ) {
                    if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += 0.014426595950721034;
                      } else {
                        result[0] += -0.008133149606088066;
                      }
                    } else {
                      result[0] += -0.03822255628484382;
                    }
                  } else {
                    if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                      result[0] += 0.013421123714351302;
                    } else {
                      result[0] += -0.06127992618619226;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.052807578792877743;
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
                      if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)12.50000000000000178) ) ) {
                        if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += 0.02384914421693706;
                        } else {
                          result[0] += 0.05204947003466335;
                        }
                      } else {
                        result[0] += -0.032964798738512505;
                      }
                    } else {
                      if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)10.50000000000000178) ) ) {
                        result[0] += -0.04775706322511756;
                      } else {
                        result[0] += 0.03514641223489755;
                      }
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.141444921493531162) ) ) {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                    result[0] += -0.14449818253634084;
                  } else {
                    result[0] += -0.004941193104483287;
                  }
                } else {
                  result[0] += 0.025153820881436896;
                }
              }
            } else {
              if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                  result[0] += -0.011462101376829013;
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.284418344497681552) ) ) {
                    result[0] += -0.011968554817367014;
                  } else {
                    result[0] += 0.06264734706538981;
                  }
                }
              } else {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
                  result[0] += -0.07207380253302838;
                } else {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += 0.01672759061098647;
                    } else {
                      if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.44140100479126021) ) ) {
                        result[0] += -0.10612735916559637;
                      } else {
                        result[0] += 0.07541808242155341;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.029068946838379794) ) ) {
                      result[0] += -0.06518177146962603;
                    } else {
                      result[0] += 0.012165179969595064;
                    }
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)3.500000000000000444) ) ) {
              if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.0029171753111309952;
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.780892848968506748) ) ) {
                      result[0] += 0.05520678090976794;
                    } else {
                      if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)6.500000000000000888) ) ) {
                        result[0] += 0.06422144522123363;
                      } else {
                        result[0] += -0.18252160787563207;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                      if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)19.50000000000000355) ) ) {
                        result[0] += 0.0228341236732834;
                      } else {
                        result[0] += -0.042698345214245065;
                      }
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
                        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                          result[0] += 0.025584955543702954;
                        } else {
                          result[0] += -0.11118013787189168;
                        }
                      } else {
                        result[0] += -0.026796949892570865;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)6.500000000000000888) ) ) {
                      if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)2.500000000000000444) ) ) {
                        result[0] += 0.05245608670779189;
                      } else {
                        result[0] += -0.058639999907794275;
                      }
                    } else {
                      if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)7.500000000000000888) ) ) {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.158952236175537998) ) ) {
                          result[0] += -0.07667234002645033;
                        } else {
                          result[0] += 0.0964992051323833;
                        }
                      } else {
                        result[0] += 0.0026519050272244033;
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
                    result[0] += 0.01724258352790709;
                  } else {
                    result[0] += -0.03337603069455601;
                  }
                } else {
                  result[0] += -0.05724356785046352;
                }
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.172047138214112216) ) ) {
                result[0] += -0.013085279040615579;
              } else {
                result[0] += 0.06203663645602812;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.284418344497681552) ) ) {
          result[0] += -0.061441054643373676;
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
            result[0] += 0.014969420671337675;
          } else {
            result[0] += 0.06902929111888607;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)40.00000000000000711) ) ) {
        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.718933820724488193) ) ) {
          result[0] += -0.017348497111334363;
        } else {
          result[0] += -0.08052929909216197;
        }
      } else {
        if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)7.000000000000000888) ) ) {
          result[0] += -0.0013177390334117639;
        } else {
          if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.718933820724488193) ) ) {
              result[0] += -0.006647576813442174;
            } else {
              result[0] += -0.05311933905375615;
            }
          } else {
            result[0] += -0.09425043343420794;
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
      if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
        result[0] += 0.05195906518616609;
      } else {
        result[0] += -0.02744059328227981;
      }
    } else {
      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += 0.02186128433603321;
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.172047138214112216) ) ) {
            result[0] += -0.03904193114249571;
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.484580039978028232) ) ) {
              result[0] += -0.0723823386572972;
            } else {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.016494801913156634;
              } else {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                  result[0] += -0.103165847594094;
                } else {
                  result[0] += 0.05537175492267019;
                }
              }
            }
          }
        }
      } else {
        result[0] += 0.038229795676209;
      }
    }
  }
  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
      if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)11.50000000000000178) ) ) {
        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)20.00000000000000355) ) ) {
          result[0] += 0.001581696648654048;
        } else {
          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)40.00000000000000711) ) ) {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.827801465988160068) ) ) {
              result[0] += -0.011185865960797958;
            } else {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)56.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.238486170768738237) ) ) {
                  result[0] += -0.010167210280673257;
                } else {
                  result[0] += -0.08168998664143762;
                }
              } else {
                result[0] += -0.014360586268135356;
              }
            }
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
                result[0] += -0.09312828014861807;
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.067782521247864214) ) ) {
                  result[0] += -0.11424031340354156;
                } else {
                  if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.02775728915022678;
                  } else {
                    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                      result[0] += 0.01841626912769031;
                    } else {
                      if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.802696108818054643) ) ) {
                        result[0] += -0.036009788541174835;
                      } else {
                        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
                          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                            result[0] += 0.009389581561647684;
                          } else {
                            result[0] += -0.04215025680635848;
                          }
                        } else {
                          result[0] += -0.08092680813408455;
                        }
                      }
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.484580039978028232) ) ) {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.02521140520285961;
                } else {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)56.00000000000000711) ) ) {
                      result[0] += 0.113350654733021;
                    } else {
                      result[0] += -0.06202733671783074;
                    }
                  } else {
                    if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                        result[0] += -0.012201258460691254;
                      } else {
                        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                          result[0] += -0.03553771641390001;
                        } else {
                          result[0] += 0.023327823858395966;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                        result[0] += 0.06461836312702743;
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
                          result[0] += 0.06364437136045414;
                        } else {
                          result[0] += -0.11837097807207136;
                        }
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
                    result[0] += 0.05334020211368867;
                  } else {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += -0.09895881469277369;
                    } else {
                      result[0] += -0.0023211214368603324;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)80.00000000000001421) ) ) {
                    if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.08141445167825946;
                    } else {
                      if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                        if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += -0.20775328928466746;
                        } else {
                          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                            result[0] += 0.1570814318871149;
                          } else {
                            result[0] += 0.04571772903128435;
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.484580039978028232) ) ) {
                          result[0] += 0.07274612374863869;
                        } else {
                          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.607751369476319248) ) ) {
                            result[0] += 0.0033546050283018154;
                          } else {
                            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                              if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)112.0000000000000142) ) ) {
                                  result[0] += 0.056255716550465754;
                                } else {
                                  result[0] += -0.1314311792544258;
                                }
                              } else {
                                result[0] += 0.04177031593234734;
                              }
                            } else {
                              result[0] += -0.06281961267188163;
                            }
                          }
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.484580039978028232) ) ) {
                        result[0] += -0.07104020800450818;
                      } else {
                        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.868834793567657693) ) ) {
                            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                              result[0] += 0.13102124502961524;
                            } else {
                              result[0] += -0.1265307372345714;
                            }
                          } else {
                            result[0] += 0.07077357665984647;
                          }
                        } else {
                          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.909855604171753818) ) ) {
                            result[0] += -0.05715811155956625;
                          } else {
                            result[0] += 0.11830489627094816;
                          }
                        }
                      }
                    } else {
                      result[0] += -0.05259229600652454;
                    }
                  }
                }
              }
            }
          }
        }
      } else {
        result[0] += -0.05614794439981942;
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.357691764831543413) ) ) {
        result[0] += -0.0732345851662733;
      } else {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)104.0000000000000142) ) ) {
              result[0] += 0.04539315059567285;
            } else {
              result[0] += -0.021086413922946187;
            }
          } else {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += -0.07453522093809188;
            } else {
              result[0] += 0.024450121451144526;
            }
          }
        } else {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.42478513717651456) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.384830474853516513) ) ) {
              result[0] += 0.13406543322745965;
            } else {
              result[0] += -0.054498105064315794;
            }
          } else {
            result[0] += -0.03407985299500817;
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
      result[0] += -0.06357137615786199;
    } else {
      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.342454433441162998) ) ) {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
          result[0] += -0.0379451736164957;
        } else {
          result[0] += 0.03134246376665044;
        }
      } else {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)2.500000000000000444) ) ) {
            result[0] += 0.07045049301425056;
          } else {
            result[0] += 0.024542550466519056;
          }
        } else {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.481347560882569248) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.347890853881836826) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.851041555404663974) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.255632162094117099) ) ) {
                    result[0] += -0.09096489666630157;
                  } else {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += -0.10058084560067382;
                    } else {
                      result[0] += 0.0005754073269938759;
                    }
                  }
                } else {
                  result[0] += 0.06455488659434602;
                }
              } else {
                result[0] += 0.06705814594465224;
              }
            } else {
              result[0] += 0.604115683795102;
            }
          } else {
            result[0] += 0.03360994868188651;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)5.500000000000000888) ) ) {
    if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)5.500000000000000888) ) ) {
      if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
        if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY(  (data[27].missing != -1) && (data[27].fvalue <= (double)-1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)257681260544.0000305) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.843275547027588779) ) ) {
                result[0] += -0.001660372041011124;
              } else {
                result[0] += 0.02692968137456372;
              }
            } else {
              if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                result[0] += 0.07117389965821645;
              } else {
                result[0] += 0.01762952640442308;
              }
            }
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)6.500000000000000888) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.1822080612182635) ) ) {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)120.0000000000000142) ) ) {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                      if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += -0.03855767400477962;
                      } else {
                        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)20.00000000000000355) ) ) {
                            result[0] += -0.03580677199061044;
                          } else {
                            result[0] += 0.019548721363042303;
                          }
                        } else {
                          result[0] += 0.02485449077506247;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.516936540603638583) ) ) {
                        result[0] += -0.0002887735938138516;
                      } else {
                        result[0] += -0.07685355989574288;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.607751369476319248) ) ) {
                          result[0] += -0.05554351011345798;
                        } else {
                          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
                            result[0] += 0.07413229011675397;
                          } else {
                            result[0] += -0.053247238650802386;
                          }
                        }
                      } else {
                        result[0] += -0.05209167668484769;
                      }
                    } else {
                      result[0] += -0.054652087716851164;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
                    if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                      if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.142630577087403232) ) ) {
                          result[0] += -0.08958908854725668;
                        } else {
                          result[0] += 0.011312659863303258;
                        }
                      } else {
                        if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                          if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)4.500000000000000888) ) ) {
                            result[0] += -0.007370015685469253;
                          } else {
                            result[0] += -0.03925025137889981;
                          }
                        } else {
                          result[0] += -0.03596863512842474;
                        }
                      }
                    } else {
                      result[0] += 0.023777098089385374;
                    }
                  } else {
                    result[0] += -0.08266226724044352;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.744781017303467685) ) ) {
                    if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                      result[0] += -0.0024123382684570825;
                    } else {
                      result[0] += -0.06986498583678596;
                    }
                  } else {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                      if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)8.500000000000001776) ) ) {
                        result[0] += 0.07031777381194258;
                      } else {
                        if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                          result[0] += -0.015534646557887133;
                        } else {
                          result[0] += 0.05814040988358743;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                        if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)7.500000000000000888) ) ) {
                          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.484580039978028232) ) ) {
                            result[0] += -0.03466386875762307;
                          } else {
                            result[0] += 0.07010528809688898;
                          }
                        } else {
                          result[0] += 0.011205163954557593;
                        }
                      } else {
                        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                          result[0] += 0.03436552189377734;
                        } else {
                          result[0] += -0.06966501184676112;
                        }
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
                    result[0] += 0.0032003829150324802;
                  } else {
                    if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.09108610327877202;
                    } else {
                      if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)7.500000000000000888) ) ) {
                        result[0] += 0.05927113927847458;
                      } else {
                        result[0] += -0.04253871317317518;
                      }
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.029068946838379794) ) ) {
                    if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
                      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                        if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                          result[0] += -0.002657445991605391;
                        } else {
                          result[0] += 0.06156666334882926;
                        }
                      } else {
                        result[0] += -0.027646272884430403;
                      }
                    } else {
                      result[0] += 0.023631228807914847;
                    }
                  } else {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.700598716735840066) ) ) {
                      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                        result[0] += 0.044711848075859664;
                      } else {
                        result[0] += -0.10629945627978535;
                      }
                    } else {
                      result[0] += -0.08309618211905631;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += -0.02789309658837821;
                  } else {
                    result[0] += 0.0025686511774394728;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.484580039978028232) ) ) {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)112.0000000000000142) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.516936540603638583) ) ) {
                      result[0] += -0.00012124811651804637;
                    } else {
                      result[0] += 0.32915910243641455;
                    }
                  } else {
                    result[0] += -0.10125120521076561;
                  }
                } else {
                  result[0] += 0.04733426915184314;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.802696108818054643) ) ) {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
              if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                result[0] += 0.01891489482373558;
              } else {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += -0.0914691705660995;
                } else {
                  result[0] += -0.018030949928905245;
                }
              }
            } else {
              result[0] += 0.022496541253677935;
            }
          } else {
            if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)4.500000000000000888) ) ) {
              result[0] += -0.04737390817662416;
            } else {
              result[0] += 0.01789573970190211;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
          result[0] += -0.01201487414491989;
        } else {
          if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.970608234405518466) ) ) {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.04940418043239289;
              } else {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.06956462442345986;
                } else {
                  result[0] += 0.03578064876277409;
                }
              }
            } else {
              result[0] += 0.11273551994852814;
            }
          } else {
            result[0] += -0.0758787262963892;
          }
        }
      }
    } else {
      result[0] += -0.06506187217302808;
    }
  } else {
    result[0] += 0.009783282441519905;
  }
  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
      if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)11.50000000000000178) ) ) {
        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)20.00000000000000355) ) ) {
          if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
            if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)6.500000000000000888) ) ) {
              if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)2.500000000000000444) ) ) {
                result[0] += -0.00031062940842812984;
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.284418344497681552) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
                    result[0] += -0.04991857086675339;
                  } else {
                    result[0] += 0.031520296258400055;
                  }
                } else {
                  if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)5.500000000000000888) ) ) {
                    result[0] += -0.07367536364170711;
                  } else {
                    result[0] += -0.024020036647322284;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
                  if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)7.500000000000000888) ) ) {
                    if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)6.500000000000000888) ) ) {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.154959201812744585) ) ) {
                        result[0] += 0.044145547296062904;
                      } else {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.827801465988160068) ) ) {
                          result[0] += 0.015028033629153332;
                        } else {
                          result[0] += -0.08429832174981113;
                        }
                      }
                    } else {
                      result[0] += 0.08736199363375843;
                    }
                  } else {
                    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                      result[0] += 0.08042743123338542;
                    } else {
                      if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.595119953155518466) ) ) {
                        result[0] += -0.10579288233871609;
                      } else {
                        result[0] += -0.0021297852003645044;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.357691764831543413) ) ) {
                    result[0] += 0.06541092565904935;
                  } else {
                    if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.07079327626753008;
                    } else {
                      if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.500000000000000444) ) ) {
                        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.357691764831543413) ) ) {
                          result[0] += 0.19147093973350987;
                        } else {
                          result[0] += -0.07350587567598961;
                        }
                      } else {
                        result[0] += 0.0036286665492861452;
                      }
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                  if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)8.500000000000001776) ) ) {
                    result[0] += 0.052163574886050494;
                  } else {
                    if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.694163918495178667) ) ) {
                        if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += -0.07508772973858849;
                        } else {
                          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            result[0] += 0.08776050085661682;
                          } else {
                            result[0] += -0.09322583128512357;
                          }
                        }
                      } else {
                        result[0] += -0.03697886724753229;
                      }
                    } else {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.90474271774292081) ) ) {
                        result[0] += 0.024443627038424905;
                      } else {
                        result[0] += -0.0712183521084993;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                    if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)8.500000000000001776) ) ) {
                      if ( LIKELY( !(data[51].missing != -1) || (data[51].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.484580039978028232) ) ) {
                          result[0] += -0.018419992712005447;
                        } else {
                          result[0] += 0.02816207100782882;
                        }
                      } else {
                        result[0] += 0.08661251060588508;
                      }
                    } else {
                      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                        result[0] += -0.03634376519981878;
                      } else {
                        result[0] += 0.015208027018182555;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.208071470260621005) ) ) {
                        result[0] += 0.050202328783690554;
                      } else {
                        if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)3.500000000000000444) ) ) {
                          result[0] += 0.02947487713581909;
                        } else {
                          result[0] += -0.05893776271873618;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.827801465988160068) ) ) {
                        result[0] += 0.006568574304488506;
                      } else {
                        result[0] += -0.06216502772489753;
                      }
                    }
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.484580039978028232) ) ) {
              result[0] += -0.09328644746835665;
            } else {
              result[0] += 0.03856908664315704;
            }
          }
        } else {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.445705175399781162) ) ) {
            if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)7.000000000000000888) ) ) {
              result[0] += 0.00032795787254021703;
            } else {
              result[0] += -0.04093940367137577;
            }
          } else {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)56.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.827801465988160068) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.516936540603638583) ) ) {
                  result[0] += -0.01697957581896762;
                } else {
                  result[0] += 0.15398276229904262;
                }
              } else {
                result[0] += -0.08351980929014687;
              }
            } else {
              result[0] += -0.008513400907735122;
            }
          }
        }
      } else {
        result[0] += -0.05355132582547783;
      }
    } else {
      if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.651049375534058505) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.827801465988160068) ) ) {
            result[0] += -0.0600761718961244;
          } else {
            result[0] += 0.06940029512759784;
          }
        } else {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.918272972106934482) ) ) {
            result[0] += 0.010410954488115997;
          } else {
            if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.10752102936271035;
            } else {
              result[0] += 0.005998263003107066;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.970608234405518466) ) ) {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.046067177435006004;
            } else {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.06668608360898819;
              } else {
                result[0] += 0.03182047353034675;
              }
            }
          } else {
            result[0] += 0.09622256233791222;
          }
        } else {
          result[0] += -0.07278661796768583;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.284418344497681552) ) ) {
      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.172047138214112216) ) ) {
        result[0] += -0.07116940741248826;
      } else {
        result[0] += -0.0030725672269430477;
      }
    } else {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.154959201812744585) ) ) {
        result[0] += -0.08763011345229177;
      } else {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)2.500000000000000444) ) ) {
            result[0] += 0.07147246260312223;
          } else {
            result[0] += 0.02200793095341305;
          }
        } else {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.90474271774292081) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.580392837524414951) ) ) {
                result[0] += -0.07465369558187467;
              } else {
                result[0] += 0.03794757429851996;
              }
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                result[0] += 0.010342435278995622;
              } else {
                result[0] += 0.20449181388755722;
              }
            }
          } else {
            result[0] += 0.026844360123618783;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
      if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)11.50000000000000178) ) ) {
        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)20.00000000000000355) ) ) {
          if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
            if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)6.500000000000000888) ) ) {
              if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)2.500000000000000444) ) ) {
                if ( UNLIKELY(  (data[27].missing != -1) && (data[27].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += 0.0024710862010512737;
                  } else {
                    if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                      result[0] += 0.0701728010986604;
                    } else {
                      result[0] += 0.01723230153265955;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)120.0000000000000142) ) ) {
                      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.909855604171753818) ) ) {
                        if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)3.500000000000000444) ) ) {
                          if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                              result[0] += 0.08538635045428075;
                            } else {
                              result[0] += -0.022470063753599268;
                            }
                          } else {
                            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
                              result[0] += -0.0007729802020332468;
                            } else {
                              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.484580039978028232) ) ) {
                                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.516936540603638583) ) ) {
                                  result[0] += -0.01396283867513671;
                                } else {
                                  result[0] += -0.1691148028129622;
                                }
                              } else {
                                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
                                  result[0] += 0.05534359643616737;
                                } else {
                                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.467917680740357333) ) ) {
                                    result[0] += -0.03812356200186144;
                                  } else {
                                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.861792564392090288) ) ) {
                                      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                                        result[0] += 0.0009732262215613918;
                                      } else {
                                        result[0] += -0.14018133623489262;
                                      }
                                    } else {
                                      result[0] += 0.042044909522903615;
                                    }
                                  }
                                }
                              }
                            }
                          }
                        } else {
                          result[0] += 0.04829264943714667;
                        }
                      } else {
                        if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)2.500000000000000444) ) ) {
                          if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)4.500000000000000888) ) ) {
                            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
                              result[0] += 0.009423898109411336;
                            } else {
                              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                                result[0] += 0.22003862950273478;
                              } else {
                                result[0] += 0.07323451515578074;
                              }
                            }
                          } else {
                            result[0] += -0.02274825825943573;
                          }
                        } else {
                          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
                            result[0] += -0.028301031294238162;
                          } else {
                            result[0] += -0.09742696111698274;
                          }
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                          result[0] += -0.017136011549024745;
                        } else {
                          result[0] += -0.05885535570010562;
                        }
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.475347042083742011) ) ) {
                          result[0] += -0.06655423457579458;
                        } else {
                          if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                            result[0] += 0.014472431335812616;
                          } else {
                            result[0] += -0.025731949189093602;
                          }
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
                      result[0] += -0.0939142310852559;
                    } else {
                      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.158952236175537998) ) ) {
                          result[0] += -0.026160852369014803;
                        } else {
                          result[0] += 0.07748560077818248;
                        }
                      } else {
                        result[0] += 0.09184577074583955;
                      }
                    }
                  }
                }
              } else {
                result[0] += -0.03439868440676827;
              }
            } else {
              result[0] += 0.005423092671611626;
            }
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.484580039978028232) ) ) {
              result[0] += -0.08421883622196236;
            } else {
              result[0] += 0.03646064554283634;
            }
          }
        } else {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.445705175399781162) ) ) {
            if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)7.000000000000000888) ) ) {
              result[0] += 9.020714984746599e-05;
            } else {
              result[0] += -0.037964332911716005;
            }
          } else {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)56.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.827801465988160068) ) ) {
                if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.994492053985595925) ) ) {
                    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.025776771762879316;
                    } else {
                      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                        result[0] += -0.0611648062738711;
                      } else {
                        result[0] += 0.013664818390359882;
                      }
                    }
                  } else {
                    result[0] += -0.07912472048378531;
                  }
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.516936540603638583) ) ) {
                    result[0] += 0.01887866526578533;
                  } else {
                    result[0] += 0.19574637384145283;
                  }
                }
              } else {
                result[0] += -0.08398935536036961;
              }
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.605039834976196733) ) ) {
                  result[0] += -0.10020441201571234;
                } else {
                  if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                    if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)7.500000000000000888) ) ) {
                      result[0] += 0.014056126505018264;
                    } else {
                      result[0] += -0.11832878697862093;
                    }
                  } else {
                    result[0] += -0.03108511162698534;
                  }
                }
              } else {
                result[0] += -0.002614104306531808;
              }
            }
          }
        }
      } else {
        result[0] += -0.0519752370205885;
      }
    } else {
      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)104.0000000000000142) ) ) {
        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.743881702423096591) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.357691764831543413) ) ) {
            result[0] += -0.058378679684476825;
          } else {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
              if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                result[0] += 0.04859871220855502;
              } else {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.08847539389442208;
                } else {
                  result[0] += 0.036171574133975294;
                }
              }
            } else {
              result[0] += 0.10291922519885598;
            }
          }
        } else {
          result[0] += -0.018269867058114512;
        }
      } else {
        result[0] += -0.022384302809369632;
      }
    }
  } else {
    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.827801465988160068) ) ) {
      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.029068946838379794) ) ) {
          result[0] += -0.07921746910911731;
        } else {
          result[0] += -0.014095605915713011;
        }
      } else {
        result[0] += 0.028612780260332343;
      }
    } else {
      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.284418344497681552) ) ) {
        result[0] += -0.014535468509349983;
      } else {
        if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)2.500000000000000444) ) ) {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += 0.07401665041660886;
          } else {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += -0.018716648044831603;
            } else {
              result[0] += 0.0409717797948532;
            }
          }
        } else {
          result[0] += 0.015251850385240498;
        }
      }
    }
  }
  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
      if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)11.50000000000000178) ) ) {
        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)20.00000000000000355) ) ) {
          if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)120.0000000000000142) ) ) {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                  result[0] += 0.002595991590879344;
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.778982400894165927) ) ) {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += -0.004011223465488439;
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.38936424255371271) ) ) {
                        result[0] += -0.04535272645857353;
                      } else {
                        if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)7.500000000000000888) ) ) {
                          if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)2.500000000000000444) ) ) {
                            result[0] += 0.050885048055673336;
                          } else {
                            result[0] += -0.02821067359432676;
                          }
                        } else {
                          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.94957673549652144) ) ) {
                            result[0] += 0.06728693560609092;
                          } else {
                            result[0] += -0.08891254444139164;
                          }
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)56.00000000000000711) ) ) {
                      if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += -0.06651275429380847;
                      } else {
                        result[0] += -0.02167379796696696;
                      }
                    } else {
                      result[0] += 0.0016780770218099623;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.038195209353893256;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.456172943115236151) ) ) {
                    result[0] += -0.08092414661165448;
                  } else {
                    if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                      result[0] += -0.023672881925442715;
                    } else {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                        result[0] += -0.013032628541483397;
                      } else {
                        result[0] += 0.03202996517995587;
                      }
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
                result[0] += -0.08892017437760548;
              } else {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.445705175399781162) ) ) {
                  result[0] += -0.043263752012080124;
                } else {
                  result[0] += 0.06536750626626633;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.484580039978028232) ) ) {
              result[0] += -0.08101125833003427;
            } else {
              result[0] += 0.032479252005073775;
            }
          }
        } else {
          if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.736135363578796831) ) ) {
              result[0] += -0.005959083558070922;
            } else {
              if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += -0.059899435102119175;
              } else {
                result[0] += -0.009485379723999854;
              }
            }
          } else {
            if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)2.500000000000000444) ) ) {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.58381557464599787) ) ) {
                  result[0] += -0.00536754277551497;
                } else {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.1083528453218686;
                  } else {
                    result[0] += -0.06799760554621276;
                  }
                }
              } else {
                result[0] += 0.022272357198095675;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.39281225204467951) ) ) {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)7.000000000000000888) ) ) {
                    if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.076962471008301669) ) ) {
                      result[0] += 0.006707138954627633;
                    } else {
                      result[0] += 0.05689468186285634;
                    }
                  } else {
                    result[0] += -0.03967716274448824;
                  }
                } else {
                  if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
                    if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)2.500000000000000444) ) ) {
                      result[0] += 0.11739672585291726;
                    } else {
                      result[0] += -0.00311979444070129;
                    }
                  } else {
                    result[0] += -0.08593399088401021;
                  }
                }
              } else {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.90474271774292081) ) ) {
                    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                      if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)7.500000000000000888) ) ) {
                        result[0] += 0.01611360716166981;
                      } else {
                        result[0] += -0.11052561522034753;
                      }
                    } else {
                      if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += -0.06416981782960435;
                      } else {
                        result[0] += 0.011034311734168946;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)72.00000000000001421) ) ) {
                      result[0] += -0.054915233195525597;
                    } else {
                      result[0] += 0.029087134997615883;
                    }
                  }
                } else {
                  result[0] += -0.00658005916262432;
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.94957673549652144) ) ) {
          result[0] += -0.0005428713353342428;
        } else {
          result[0] += -0.06635389990274891;
        }
      }
    } else {
      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)104.0000000000000142) ) ) {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.48918962478637873) ) ) {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.42478513717651456) ) ) {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
              if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += -0.07245002555910994;
              } else {
                if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                  result[0] += 0.049848699843620795;
                } else {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += -0.05444683153431043;
                  } else {
                    result[0] += 0.056929632018580735;
                  }
                }
              }
            } else {
              result[0] += 0.099317615734322;
            }
          } else {
            result[0] += -0.010605516400368827;
          }
        } else {
          result[0] += -0.09780530748660497;
        }
      } else {
        result[0] += -0.022018523383502978;
      }
    }
  } else {
    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.154959201812744585) ) ) {
      result[0] += -0.08742653457316825;
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.02604460716247603) ) ) {
        result[0] += -0.11875086621432392;
      } else {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)15.50000000000000178) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.947818994522095615) ) ) {
              result[0] += 0.05025136754514196;
            } else {
              result[0] += -0.08295812508423303;
            }
          } else {
            result[0] += -0.06072781434640223;
          }
        } else {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.743881702423096591) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.42478513717651456) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.993164777755738193) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.433569431304932529) ) ) {
                    result[0] += -0.11754321190583886;
                  } else {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += -0.0822969589904416;
                    } else {
                      result[0] += 0.009947375581135927;
                    }
                  }
                } else {
                  result[0] += -0.006507069285888143;
                }
              } else {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += -0.07136385743419832;
                } else {
                  result[0] += 0.02768175741763368;
                }
              }
            } else {
              result[0] += 0.09307950043397512;
            }
          } else {
            result[0] += 0.029794499564537086;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
      if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)11.50000000000000178) ) ) {
        if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
          if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.651049375534058505) ) ) {
                if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)13.50000000000000178) ) ) {
                  if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)4.500000000000000888) ) ) {
                    if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)2.500000000000000444) ) ) {
                      if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)1.500000000000000222) ) ) {
                        if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += 0.0011330424765207997;
                        } else {
                          result[0] += 0.03259228151527172;
                        }
                      } else {
                        result[0] += -0.11223866348511935;
                      }
                    } else {
                      result[0] += 0.04479897990715965;
                    }
                  } else {
                    result[0] += -0.020332037174648692;
                  }
                } else {
                  if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)8.500000000000001776) ) ) {
                    if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)10.50000000000000178) ) ) {
                      result[0] += 0.06347352662202393;
                    } else {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.516936540603638583) ) ) {
                        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.361115694046021396) ) ) {
                            result[0] += 0.08562578379907444;
                          } else {
                            result[0] += -0.09627370372663582;
                          }
                        } else {
                          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.95478391647339045) ) ) {
                            result[0] += 0.02536049096585019;
                          } else {
                            result[0] += -0.02018949957601873;
                          }
                        }
                      } else {
                        result[0] += -0.0463250599877009;
                      }
                    }
                  } else {
                    result[0] += -0.03867508151095145;
                  }
                }
              } else {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                    result[0] += 0.015531680762410478;
                  } else {
                    if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)5.500000000000000888) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.040716171264650214) ) ) {
                        result[0] += 0.04726727938260334;
                      } else {
                        result[0] += -0.011954296280060103;
                      }
                    } else {
                      result[0] += -0.05905334183952508;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                    if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.208071470260621005) ) ) {
                        result[0] += -0.03317114015533248;
                      } else {
                        result[0] += -0.0983902740637706;
                      }
                    } else {
                      if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)6.500000000000000888) ) ) {
                        result[0] += -0.022248641243910336;
                      } else {
                        if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.500000000000000444) ) ) {
                          result[0] += 0.09563170786926106;
                        } else {
                          if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)7.500000000000000888) ) ) {
                            result[0] += 0.05347304854639905;
                          } else {
                            result[0] += -0.004849118460143778;
                          }
                        }
                      }
                    }
                  } else {
                    result[0] += -0.058633288173010605;
                  }
                }
              }
            } else {
              result[0] += -0.06310758448384936;
            }
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.484580039978028232) ) ) {
              result[0] += -0.037698926013153514;
            } else {
              if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                  result[0] += 0.017598658765142487;
                } else {
                  result[0] += -0.044520255415426375;
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.671854496002199042) ) ) {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += 0.007447148697027628;
                  } else {
                    result[0] += -0.13712188377136372;
                  }
                } else {
                  result[0] += 0.03661843408303117;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.605039834976196733) ) ) {
              result[0] += -0.10152854918157078;
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.605039834976196733) ) ) {
                result[0] += -0.06080201247265038;
              } else {
                if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)40.00000000000000711) ) ) {
                    result[0] += -0.09203406104491535;
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.909855604171753818) ) ) {
                      result[0] += 0.006443043823328264;
                    } else {
                      result[0] += -0.03214127052116266;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)2.500000000000000444) ) ) {
                    result[0] += -0.10055218931052051;
                  } else {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.516936540603638583) ) ) {
                      if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)7.500000000000000888) ) ) {
                        result[0] += 0.004299520737366734;
                      } else {
                        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
                          result[0] += 0.02850606702485315;
                        } else {
                          result[0] += -0.049401549257688554;
                        }
                      }
                    } else {
                      result[0] += -0.00651869731886853;
                    }
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.778982400894165927) ) ) {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += 0.011983736223330807;
                } else {
                  result[0] += 0.052308723682539665;
                }
              } else {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)56.00000000000000711) ) ) {
                  result[0] += -0.07731944901869252;
                } else {
                  result[0] += 0.02896561802661317;
                }
              }
            } else {
              if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)80.00000000000001421) ) ) {
                if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.605039834976196733) ) ) {
                  if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.802696108818054643) ) ) {
                      if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
                        result[0] += -0.006947337564900319;
                      } else {
                        result[0] += 0.0962593086799633;
                      }
                    } else {
                      if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)3.500000000000000444) ) ) {
                        result[0] += -0.06772321866517682;
                      } else {
                        result[0] += -0.013649957937022315;
                      }
                    }
                  } else {
                    result[0] += 0.0177829904107218;
                  }
                } else {
                  result[0] += -0.034353411493535616;
                }
              } else {
                result[0] += 0.00022234508601982935;
              }
            }
          }
        }
      } else {
        result[0] += -0.04670165383418101;
      }
    } else {
      if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
        result[0] += 0.10370674732691083;
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.02604460716247603) ) ) {
          result[0] += -0.12530787222635958;
        } else {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)104.0000000000000142) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.90474271774292081) ) ) {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.445705175399781162) ) ) {
                result[0] += -0.02728702881832994;
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.154959201812744585) ) ) {
                  result[0] += -0.080712523300062;
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.43742394447326749) ) ) {
                    result[0] += 0.05599141708815772;
                  } else {
                    result[0] += -0.0369046094190985;
                  }
                }
              }
            } else {
              result[0] += -0.02531264923144962;
            }
          } else {
            result[0] += -0.01899720883103389;
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.342454433441162998) ) ) {
      result[0] += -0.0206172231232061;
    } else {
      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.802696108818054643) ) ) {
        result[0] += -0.04139576337412635;
      } else {
        result[0] += 0.024327288455197882;
      }
    }
  }
  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
        result[0] += 0.005281216042292808;
      } else {
        result[0] += 0.027541682310036703;
      }
    } else {
      if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
        if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)11.50000000000000178) ) ) {
          if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)6.500000000000000888) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.1822080612182635) ) ) {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)120.0000000000000142) ) ) {
                  if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.03348879408034509;
                  } else {
                    if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                      if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.039720773696899636) ) ) {
                        result[0] += 0.013163533965513483;
                      } else {
                        result[0] += -0.06810838470982863;
                      }
                    } else {
                      result[0] += 0.011230130682700873;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.607751369476319248) ) ) {
                        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += 0.05713579811767312;
                        } else {
                          result[0] += -0.08306651525246637;
                        }
                      } else {
                        result[0] += 0.05360323658816729;
                      }
                    } else {
                      result[0] += -0.048230433095861266;
                    }
                  } else {
                    result[0] += -0.050073482570682004;
                  }
                }
              } else {
                if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                  if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.802696108818054643) ) ) {
                      if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += -0.08413697443331335;
                      } else {
                        if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)4.500000000000000888) ) ) {
                          result[0] += 0.06137091503080096;
                        } else {
                          result[0] += -0.020336238164517025;
                        }
                      }
                    } else {
                      result[0] += -0.023673354578118908;
                    }
                  } else {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.433569431304932529) ) ) {
                      if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                        result[0] += 0.029310549833484204;
                      } else {
                        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                          result[0] += -0.09060644857959922;
                        } else {
                          result[0] += -0.010865028414241328;
                        }
                      }
                    } else {
                      result[0] += 0.015789997633803458;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.04726597770467083;
                  } else {
                    result[0] += -0.0018068516491393698;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.029068946838379794) ) ) {
                  if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += -0.008337910315134712;
                  } else {
                    result[0] += 0.022855746085967757;
                  }
                } else {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.700598716735840066) ) ) {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += 0.027215848325270006;
                    } else {
                      result[0] += -0.13099438199863145;
                    }
                  } else {
                    result[0] += -0.07763551100777724;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.48918962478637873) ) ) {
                      result[0] += -0.0035741538479160057;
                    } else {
                      if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
                        result[0] += 0.06340090476665759;
                      } else {
                        if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += 0.030275425660525546;
                        } else {
                          result[0] += -0.04208897871406461;
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.827801465988160068) ) ) {
                      result[0] += -0.0615011305825431;
                    } else {
                      if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                        if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.718933820724488193) ) ) {
                            result[0] += -0.10955282386500526;
                          } else {
                            result[0] += 0.07257435191287306;
                          }
                        } else {
                          result[0] += -0.021782250928076542;
                        }
                      } else {
                        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.012675821781158891) ) ) {
                            result[0] += 0.09326060323224038;
                          } else {
                            result[0] += -0.0008730585885367425;
                          }
                        } else {
                          result[0] += 0.13145669711676228;
                        }
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.938867926597595659) ) ) {
                    result[0] += -0.07104001561491084;
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.79870843887329279) ) ) {
                      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)40.00000000000000711) ) ) {
                        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.778982400894165927) ) ) {
                          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.802696108818054643) ) ) {
                            result[0] += 0.10229096876013964;
                          } else {
                            result[0] += 0.03533883166054408;
                          }
                        } else {
                          result[0] += -0.06833669273808478;
                        }
                      } else {
                        if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.44140100479126021) ) ) {
                          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)80.00000000000001421) ) ) {
                            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.302512168884278232) ) ) {
                              result[0] += 0.00901811864675451;
                            } else {
                              result[0] += -0.028862530366774698;
                            }
                          } else {
                            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.484580039978028232) ) ) {
                              result[0] += -0.04452155269654683;
                            } else {
                              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                                result[0] += 0.05177250618260257;
                              } else {
                                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.861792564392090288) ) ) {
                                    result[0] += -0.09059885034838228;
                                  } else {
                                    result[0] += 0.03242384900381863;
                                  }
                                } else {
                                  result[0] += -0.07390970164986176;
                                }
                              }
                            }
                          }
                        } else {
                          if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)3.500000000000000444) ) ) {
                            result[0] += -0.01969883856648736;
                          } else {
                            result[0] += -0.13328985429251639;
                          }
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                        result[0] += -0.05438375554445242;
                      } else {
                        result[0] += 0.0366774144582351;
                      }
                    }
                  }
                }
              }
            }
          } else {
            result[0] += 0.004009185830511333;
          }
        } else {
          result[0] += -0.045614978404157465;
        }
      } else {
        if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
          result[0] += -0.01102263242767682;
        } else {
          result[0] += 0.0325163046859074;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.802696108818054643) ) ) {
      result[0] += -0.030544727772347582;
    } else {
      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.802696108818054643) ) ) {
        result[0] += -0.03936289083229341;
      } else {
        if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
          result[0] += 0.04373379754278124;
        } else {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.780892848968506748) ) ) {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.56941866874694913) ) ) {
                result[0] += -0.07074438387476928;
              } else {
                result[0] += 0.004882628350131352;
              }
            } else {
              result[0] += 0.08393168080832825;
            }
          } else {
            result[0] += 0.026991928172422627;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY(  (data[27].missing != -1) && (data[27].fvalue <= (double)-1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
        result[0] += 0.00509182456999531;
      } else {
        result[0] += 0.03169300234613903;
      }
    } else {
      if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
        if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)11.50000000000000178) ) ) {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.671854496002199042) ) ) {
                  result[0] += -0.0378204199849674;
                } else {
                  if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)8.500000000000001776) ) ) {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)72.00000000000001421) ) ) {
                      result[0] += 0.024598266312543287;
                    } else {
                      if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += 0.035996687825678594;
                      } else {
                        result[0] += -0.06205862817091491;
                      }
                    }
                  } else {
                    result[0] += 0.08849550723565885;
                  }
                }
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.142747402191162998) ) ) {
                  if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.500000000000000444) ) ) {
                    if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                      if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.694163918495178667) ) ) {
                          result[0] += 0.036615048938769615;
                        } else {
                          result[0] += -0.05097881821787137;
                        }
                      } else {
                        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)4.500000000000000888) ) ) {
                          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                            result[0] += 0.0016825407016092674;
                          } else {
                            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.918272972106934482) ) ) {
                              result[0] += -0.03137642009615415;
                            } else {
                              result[0] += -0.10617681378242112;
                            }
                          }
                        } else {
                          result[0] += 0.006659778485571252;
                        }
                      }
                    } else {
                      result[0] += 0.024380214704229262;
                    }
                  } else {
                    if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)24.50000000000000355) ) ) {
                      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                        result[0] += 0.03617613140230932;
                      } else {
                        result[0] += -0.0296769532877028;
                      }
                    } else {
                      result[0] += -0.022317406674807708;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)13.50000000000000178) ) ) {
                    result[0] += -0.02744005132752916;
                  } else {
                    result[0] += -0.12526984395301338;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.172047138214112216) ) ) {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                  if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                      result[0] += -0.09929282755461144;
                    } else {
                      if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += 0.09111131595259404;
                      } else {
                        result[0] += -0.033440970067987956;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.173939466476441318) ) ) {
                      result[0] += -0.09535520698585209;
                    } else {
                      result[0] += 0.10286190951317127;
                    }
                  }
                } else {
                  result[0] += -0.0839485242649689;
                }
              } else {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.700598716735840066) ) ) {
                  if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                    result[0] += 0.005236786827430965;
                  } else {
                    result[0] += 0.13250685088385403;
                  }
                } else {
                  if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.90474271774292081) ) ) {
                      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)72.00000000000001421) ) ) {
                          if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                              result[0] += 0.03161165237107062;
                            } else {
                              result[0] += -0.052356766760877194;
                            }
                          } else {
                            result[0] += -0.07645721328512643;
                          }
                        } else {
                          result[0] += -0.09243225258069027;
                        }
                      } else {
                        result[0] += 0.0029566456874699993;
                      }
                    } else {
                      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                        result[0] += 0.09372947460321271;
                      } else {
                        result[0] += -0.034966094459062264;
                      }
                    }
                  } else {
                    result[0] += -0.09961394694851125;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.894675970077515537) ) ) {
              result[0] += -0.05231472731536452;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.00047874450683771) ) ) {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.827801465988160068) ) ) {
                    result[0] += 0.014067520167815607;
                  } else {
                    result[0] += -0.07294292249448425;
                  }
                } else {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                    result[0] += -0.1262174172150682;
                  } else {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.449861526489258257) ) ) {
                      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.516936540603638583) ) ) {
                        result[0] += -0.01914387400062397;
                      } else {
                        result[0] += -0.10086681079079332;
                      }
                    } else {
                      result[0] += 0.0389221643718965;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += -0.0693018116514166;
                    } else {
                      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                        result[0] += 0.03014064489103819;
                      } else {
                        if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.595119953155518466) ) ) {
                          result[0] += -0.023014637895865515;
                        } else {
                          result[0] += -0.0772369339779265;
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)5.500000000000000888) ) ) {
                        if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                            result[0] += -0.03674560455829104;
                          } else {
                            result[0] += 0.01373102200099352;
                          }
                        } else {
                          result[0] += -0.0019588487089748333;
                        }
                      } else {
                        result[0] += 0.007670340107398758;
                      }
                    } else {
                      if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += -0.05614804278593627;
                      } else {
                        result[0] += 0.018905350152941952;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.284418344497681552) ) ) {
                    result[0] += 0.024023944643734214;
                  } else {
                    if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                      result[0] += -0.08341610284479553;
                    } else {
                      if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                        if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)7.500000000000000888) ) ) {
                          if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)6.500000000000000888) ) ) {
                            result[0] += 0.00047243319221485663;
                          } else {
                            result[0] += 0.0692770257382062;
                          }
                        } else {
                          result[0] += -0.055478356035631275;
                        }
                      } else {
                        result[0] += -0.0862513175633431;
                      }
                    }
                  }
                }
              }
            }
          }
        } else {
          result[0] += -0.04322622471163251;
        }
      } else {
        if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
          result[0] += -0.010674293028106668;
        } else {
          result[0] += 0.030178834649950827;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.154959201812744585) ) ) {
      result[0] += -0.08255031105409633;
    } else {
      result[0] += 0.017500293854306535;
    }
  }
  if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)16.50000000000000355) ) ) {
    if ( UNLIKELY(  (data[27].missing != -1) && (data[27].fvalue <= (double)-1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
        result[0] += 0.004919145216849302;
      } else {
        if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
          result[0] += 0.06839520173568646;
        } else {
          result[0] += 0.018007946796554573;
        }
      }
    } else {
      if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
                  result[0] += 0.025866045479475117;
                } else {
                  result[0] += -0.0564788608043148;
                }
              } else {
                result[0] += -0.008443306215638142;
              }
            } else {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)4.500000000000000888) ) ) {
                  if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.500000000000000444) ) ) {
                    if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.124530076980591708) ) ) {
                      if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
                        result[0] += -0.014993049673529896;
                      } else {
                        if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.802696108818054643) ) ) {
                          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.515218973159790483) ) ) {
                            result[0] += 0.10787956468565314;
                          } else {
                            result[0] += -0.05568148989325472;
                          }
                        } else {
                          result[0] += 0.036150692500533814;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
                        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                          if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)2.500000000000000444) ) ) {
                            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
                              result[0] += 0.13849205752004853;
                            } else {
                              result[0] += -0.02150562388826638;
                            }
                          } else {
                            result[0] += -0.09136681985392524;
                          }
                        } else {
                          result[0] += -0.10748887937759882;
                        }
                      } else {
                        result[0] += -0.012788157019535736;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)8.500000000000001776) ) ) {
                      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.450390577316285068) ) ) {
                        result[0] += 0.03489867301744391;
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.172047138214112216) ) ) {
                          result[0] += 0.0300691644332554;
                        } else {
                          result[0] += -0.05348862236513933;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.284418344497681552) ) ) {
                        if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)30.50000000000000355) ) ) {
                          result[0] += -0.04496934029161361;
                        } else {
                          result[0] += -0.23319776713582702;
                        }
                      } else {
                        if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)26.50000000000000355) ) ) {
                          result[0] += -0.06997434621742792;
                        } else {
                          result[0] += 0.01556193123526361;
                        }
                      }
                    }
                  }
                } else {
                  result[0] += 0.0868155239387457;
                }
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
                  result[0] += -0.1079364573447773;
                } else {
                  result[0] += 0.07831902285864367;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.516936540603638583) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.705447435379029208) ) ) {
                if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.07057978124435103;
                } else {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.357691764831543413) ) ) {
                      result[0] += 0.1373987095246055;
                    } else {
                      result[0] += -0.007632569533941792;
                    }
                  } else {
                    if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
                      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                        result[0] += -0.037682415773683;
                      } else {
                        result[0] += 0.11343273282217564;
                      }
                    } else {
                      result[0] += -0.07471959244367431;
                    }
                  }
                }
              } else {
                result[0] += 0.05302588076273268;
              }
            } else {
              if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.172047138214112216) ) ) {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)168.0000000000000284) ) ) {
                    result[0] += -0.09147630281350252;
                  } else {
                    result[0] += 0.08445088042191538;
                  }
                } else {
                  if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
                    result[0] += -0.09127353929772805;
                  } else {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)72.00000000000001421) ) ) {
                      if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
                        result[0] += 0.008464372603970542;
                      } else {
                        result[0] += -0.0924311070233848;
                      }
                    } else {
                      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)46.00000000000000711) ) ) {
                        result[0] += 0.03796433555134191;
                      } else {
                        result[0] += -0.05935348139011605;
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
                      result[0] += 0.09399072954180387;
                    } else {
                      result[0] += -0.0898275475412914;
                    }
                  } else {
                    result[0] += 0.09005309544761646;
                  }
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.827801465988160068) ) ) {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.012675821781158891) ) ) {
                      result[0] += 0.08832453360602041;
                    } else {
                      if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.44140100479126021) ) ) {
                        result[0] += -0.02302067928004495;
                      } else {
                        result[0] += 0.12377381173446794;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)7.500000000000000888) ) ) {
                      result[0] += 0.06461320945177561;
                    } else {
                      result[0] += -0.0718115055384781;
                    }
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)2.65906000137329146) ) ) {
            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.938867926597595659) ) ) {
              result[0] += -0.06135851524010565;
            } else {
              if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.827801465988160068) ) ) {
                  result[0] += 0.002074032330951471;
                } else {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                    result[0] += 0.021772746347112275;
                  } else {
                    result[0] += -0.014974873636357904;
                  }
                }
              } else {
                result[0] += 0.0047917105571987206;
              }
            }
          } else {
            result[0] += -0.16777424165924087;
          }
        }
      } else {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
          if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
            result[0] += 0.02137681835501793;
          } else {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += -0.06802439816840619;
            } else {
              result[0] += 0.0060317076278026995;
            }
          }
        } else {
          result[0] += 0.0551309297291686;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
      result[0] += -0.005520687982230644;
    } else {
      if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)10.50000000000000178) ) ) {
        result[0] += 0.061427123472111635;
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.827801465988160068) ) ) {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
            result[0] += -0.027624104533457164;
          } else {
            result[0] += 0.029528301537403212;
          }
        } else {
          if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)2.500000000000000444) ) ) {
            result[0] += 0.04387044402359566;
          } else {
            result[0] += 0.00859883639733057;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
        result[0] += 0.004563019245504423;
      } else {
        result[0] += 0.0258208953906521;
      }
    } else {
      if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
        if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)11.50000000000000178) ) ) {
          if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)6.500000000000000888) ) ) {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)80.00000000000001421) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.484580039978028232) ) ) {
                if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += 0.01009629585336659;
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.694163918495178667) ) ) {
                      if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)2.500000000000000444) ) ) {
                        result[0] += 0.09896454452706248;
                      } else {
                        result[0] += -0.05893030462453727;
                      }
                    } else {
                      result[0] += 0.011833104668167106;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += 0.032586684539571885;
                  } else {
                    if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.970608234405518466) ) ) {
                      result[0] += -0.009040431548799356;
                    } else {
                      if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)4.500000000000000888) ) ) {
                        result[0] += -0.015932550941092458;
                      } else {
                        result[0] += -0.06404365050025775;
                      }
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.780892848968506748) ) ) {
                      result[0] += -0.008167506214320582;
                    } else {
                      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                        result[0] += 0.15047775379407344;
                      } else {
                        result[0] += 0.033145159052820804;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.500000000000000444) ) ) {
                      result[0] += -0.06962070220258143;
                    } else {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.849175214767456943) ) ) {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.285887241363526279) ) ) {
                          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.029068946838379794) ) ) {
                            if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
                              result[0] += 0.019674822004597352;
                            } else {
                              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
                                result[0] += -0.04657583997375844;
                              } else {
                                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                                  result[0] += -0.0599195747952421;
                                } else {
                                  if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                                    result[0] += 0.01982833064620101;
                                  } else {
                                    result[0] += -0.05374451389470996;
                                  }
                                }
                              }
                            }
                          } else {
                            result[0] += 0.010343336238825517;
                          }
                        } else {
                          result[0] += -0.05770785373972268;
                        }
                      } else {
                        result[0] += -0.048820271673125415;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                    if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)3.500000000000000444) ) ) {
                      result[0] += -0.09896443530628862;
                    } else {
                      result[0] += -0.04586548766587539;
                    }
                  } else {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.051912069320679599) ) ) {
                      result[0] += 0.0047052970689675415;
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
                        result[0] += -0.002174560768969866;
                      } else {
                        result[0] += -0.07249859018824088;
                      }
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                  if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                    result[0] += -0.013957323676478903;
                  } else {
                    result[0] += -0.06133599467316067;
                  }
                } else {
                  result[0] += 0.08044603282987262;
                }
              } else {
                if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.142630577087403232) ) ) {
                    result[0] += -0.0429526493751293;
                  } else {
                    if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                      if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                        result[0] += 0.025875467648503465;
                      } else {
                        result[0] += -0.047433480224367434;
                      }
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.205872535705568183) ) ) {
                        result[0] += -0.02013447857508372;
                      } else {
                        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.142630577087403232) ) ) {
                          result[0] += -0.02598916755195306;
                        } else {
                          result[0] += 0.03252581836664619;
                        }
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.605039834976196733) ) ) {
                      result[0] += -0.08972461232984781;
                    } else {
                      if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)7.000000000000000888) ) ) {
                        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.605039834976196733) ) ) {
                          result[0] += -0.05497545269859801;
                        } else {
                          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
                            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.718933820724488193) ) ) {
                              result[0] += 0.00834796060449696;
                            } else {
                              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                                result[0] += -0.07441591674632778;
                              } else {
                                result[0] += -0.0051864504085262315;
                              }
                            }
                          } else {
                            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.90474271774292081) ) ) {
                              result[0] += -0.06270422019716834;
                            } else {
                              result[0] += 0.022971082747899864;
                            }
                          }
                        }
                      } else {
                        result[0] += -0.027961168347479488;
                      }
                    }
                  } else {
                    result[0] += 0.0028947123350911612;
                  }
                }
              }
            }
          } else {
            result[0] += 0.003192774065029982;
          }
        } else {
          result[0] += -0.04183280720163835;
        }
      } else {
        if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.970608234405518466) ) ) {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.023191882458427945;
            } else {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.0663821953372659;
              } else {
                result[0] += 0.009344263424858598;
              }
            }
          } else {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.42478513717651456) ) ) {
              result[0] += 0.07067524342340586;
            } else {
              result[0] += -0.03605570760268181;
            }
          }
        } else {
          result[0] += -0.07463512509795982;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.172047138214112216) ) ) {
      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.339395284652710849) ) ) {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += -0.0046085850871867414;
          } else {
            result[0] += -0.11030043730465706;
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.094205617904663974) ) ) {
            result[0] += -0.09527981734801107;
          } else {
            result[0] += 0.044064392548955444;
          }
        }
      } else {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += 0.058131025429264106;
        } else {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += -0.05460819254770566;
          } else {
            result[0] += 0.024118667654559344;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)2.500000000000000444) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.029068946838379794) ) ) {
          result[0] += 0.016535346893329618;
        } else {
          result[0] += 0.05813005903073703;
        }
      } else {
        result[0] += 0.012657006656688992;
      }
    }
  }
  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)257681260544.0000305) ) ) {
        if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
          if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.088880300521851474) ) ) {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.607751369476319248) ) ) {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.02620365797943261;
                } else {
                  result[0] += -0.007457967961831386;
                }
              } else {
                result[0] += -0.10109084789878833;
              }
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.924581527709961826) ) ) {
                result[0] += 0.0783095997178143;
              } else {
                result[0] += -6.249958660882872e-05;
              }
            }
          } else {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.718933820724488193) ) ) {
              result[0] += -0.10277143186328366;
            } else {
              result[0] += 0.0242273009359945;
            }
          }
        } else {
          result[0] += 0.010698768682068018;
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.055836200714113104) ) ) {
          result[0] += -0.0005490442284427783;
        } else {
          result[0] += 0.03670373810369852;
        }
      }
    } else {
      if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
        if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)11.50000000000000178) ) ) {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.790835380554201) ) ) {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.01352395361216491;
              } else {
                result[0] += 0.009640327730741523;
              }
            } else {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.029068946838379794) ) ) {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)88.00000000000001421) ) ) {
                    result[0] += 0.018589672102930267;
                  } else {
                    result[0] += -0.02855873429389256;
                  }
                } else {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                    result[0] += -0.019894063656208667;
                  } else {
                    result[0] += 0.07258557505685233;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += -0.06397501300740556;
                  } else {
                    if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
                      if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += -0.044363195487875436;
                      } else {
                        result[0] += -0.001978488406569439;
                      }
                    } else {
                      result[0] += 0.02857354708585676;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.04236639259421901;
                  } else {
                    result[0] += 0.000556305248843356;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.172047138214112216) ) ) {
              if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.48918962478637873) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.7512402534484881) ) ) {
                    result[0] += -0.13236922874081974;
                  } else {
                    result[0] += -0.04191085649824546;
                  }
                } else {
                  result[0] += 0.04799487847922418;
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.055836200714113104) ) ) {
                  if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                    result[0] += -0.03932729350799564;
                  } else {
                    result[0] += 0.12361349898462058;
                  }
                } else {
                  result[0] += 0.022154336343224505;
                }
              }
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.802901029586792436) ) ) {
                result[0] += -0.09588754342979275;
              } else {
                if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                  result[0] += -0.07911578445191947;
                } else {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.00861720815226395;
                  } else {
                    result[0] += -0.08386209349072918;
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.484580039978028232) ) ) {
            result[0] += -0.0003287409576359305;
          } else {
            result[0] += -0.06625683542074;
          }
        }
      } else {
        if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.651049375534058505) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.827801465988160068) ) ) {
              result[0] += -0.05908859661896779;
            } else {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
                result[0] += 0.07589590018120533;
              } else {
                result[0] += -0.06462856092155667;
              }
            }
          } else {
            result[0] += -0.03544767640856712;
          }
        } else {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)104.0000000000000142) ) ) {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.484580039978028232) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.094205617904663974) ) ) {
                  result[0] += -0.06983391358087958;
                } else {
                  result[0] += 0.07696880071521547;
                }
              } else {
                result[0] += -0.03159389060328142;
              }
            } else {
              if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)4.500000000000000888) ) ) {
                if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                  result[0] += -0.012930395639620576;
                } else {
                  result[0] += 0.07894958838149749;
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.694163918495178667) ) ) {
                  result[0] += -0.021710083651013803;
                } else {
                  result[0] += 0.07968677819229658;
                }
              }
            }
          } else {
            result[0] += -0.03612541821798136;
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.154959201812744585) ) ) {
      result[0] += -0.07761303816488686;
    } else {
      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)15.50000000000000178) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.947818994522095615) ) ) {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
              result[0] += 0.08216383232742572;
            } else {
              if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)2.500000000000000444) ) ) {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.010212063858184715;
                  } else {
                    result[0] += 0.0832415395449557;
                  }
                } else {
                  if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.534971714019776279) ) ) {
                    result[0] += 0.032552335017738084;
                  } else {
                    if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)17.50000000000000355) ) ) {
                      result[0] += 0.02633678068675508;
                    } else {
                      result[0] += -0.12949938014787774;
                    }
                  }
                }
              } else {
                result[0] += 0.09074446589873941;
              }
            }
          } else {
            result[0] += -0.0775629352256024;
          }
        } else {
          result[0] += -0.0616781605782777;
        }
      } else {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.90474271774292081) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.580392837524414951) ) ) {
              result[0] += -0.06538571819132444;
            } else {
              result[0] += 0.026436483124297444;
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.070054531097412998) ) ) {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += -0.0711225903273173;
              } else {
                result[0] += 0.02673500513326151;
              }
            } else {
              result[0] += 0.13714089819443193;
            }
          }
        } else {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.802696108818054643) ) ) {
            result[0] += -0.049633810200081646;
          } else {
            result[0] += 0.021532006716637672;
          }
        }
      }
    }
  }
}

