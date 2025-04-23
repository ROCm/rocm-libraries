#-------------------------------------------------
#
# Project created by QtCreator 2017-10-10T16:38:59
#
#-------------------------------------------------

QT     += core gui network

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets printsupport

# With C++11 support
CONFIG += c++11

TARGET = annInferenceApp
TEMPLATE = app

# The following define makes your compiler emit warnings if you use
# any feature of Qt which as been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
        main.cpp \
        inference_viewer.cpp \
        inference_receiver.cpp \
        inference_control.cpp \
        inference_compiler.cpp \
        assets.cpp \
        tcpconnection.cpp \
        inference_panel.cpp \
        perf_graph.cpp \
        perf_chart.cpp \
        qcustomplot.cpp

HEADERS += \
        inference_viewer.h \
        inference_receiver.h \
        inference_control.h \
        inference_compiler.h \
        infcom.h \
        assets.h \
        tcpconnection.h \
        inference_panel.h \
        perf_graph.h \
        perf_chart.h \
        qcustomplot.h

FORMS += \
        inference_viewer.ui \
        inference_panel.ui \
        perf_graph.ui \
        perf_chart.ui

RESOURCES += \
    resources.qrc
