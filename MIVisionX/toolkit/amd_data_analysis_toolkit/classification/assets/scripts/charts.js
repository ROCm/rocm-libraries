/*global google, data, $, modelHistories*/

function sum(a, b) {
    return a + b;
}


function drawChart() {

    var chartData = google.visualization.arrayToDataTable([
        ["  ", "Match", "Mismatch", "No Label"],
        ["Summary", data.stats.passCount, data.stats.totalMismatch, data.stats.totalNoGroundTruth]
    ]);
    var options = {
        title: "Overall Result Summary",
        vAxis: {
            title: "Images"
        },
        width: 800,
        height: 400
    };
    var chart = new google.charts.Bar(document.getElementById("Model_Stats"));
    chart.draw(chartData, google.charts.Bar.convertOptions(options));
}


function drawTopKResultChart() {
    var chartData = new google.visualization.DataTable();
    chartData.addColumn("string", "Top K");
    chartData.addColumn("number", "Matchs");
    chartData.addRows([
        ["Matched Top5 Choice", data.topCounts.reduce(sum, 0)],
        ["MisMatched", data.stats.totalMismatch]
    ]);
    var options = {
        title: "Image Match/Mismatch Summary",
        width: 750,
        height: 400
    };
    var chart = new google.visualization.PieChart(document.getElementById("topK_result_chart_div"));
    chart.draw(chartData, options);
}


function drawResultChart() {
    var chartData = new google.visualization.DataTable();
    chartData.addColumn("string", "Top K");
    chartData.addColumn("number", "Matchs");
    chartData.addRows([
        ["Matched 1st Choice", data.topCounts[0]],
        ["Matched 2nd Choice", data.topCounts[1]],
        ["Matched 3rd Choice", data.topCounts[2]],
        ["Matched 4th Choice", data.topCounts[3]],
        ["Matched 5th Choice", data.topCounts[4]]
    ]);
    var options = {
        title: "Image Matches",
        width: 750,
        height: 400
    };
    var chart = new google.visualization.PieChart(document.getElementById("result_chart_div"));
    chart.draw(chartData, options);
}


function drawPassFailGraph() {

    var chartParentDiv = document.createElement("div");
    chartParentDiv.className = "column";

    var chartDiv = document.createElement("div");
    chartDiv.classList.add("pass-fail-chart");



    var chartData = new google.visualization.DataTable();
    chartData.addColumn("number", "X");
    chartData.addColumn("number", "Match");
    chartData.addColumn("number", "Mismatch");
    chartData.addRows(data.chartData.passFailData);
    var options = {
        title: "Cummulative Success/Failure",
        hAxis: {
            title: "Confidence",
            direction: "-1"
        },
        vAxis: {
            title: "Percentage of Dataset"
        },
        series: {
            0.01: {
                curveType: "function"
            }
        },
        width: 700,
        height: 400
    };
    var chart = new google.visualization.LineChart(chartDiv);
    chart.draw(chartData, options);
    chartParentDiv.appendChild(chartDiv);
    document.getElementById("passFailGraphs").appendChild(chartParentDiv);
}


function drawLnPassFailGraphs() {

    for (var i = 0; i < data.chartData.lnPassFailData.length; i++) {

        // There is no L6 so do not plot L6 SUccess Failure
        if (i === 5)
            continue;

        var chartParentDiv = document.createElement("div");
        chartParentDiv.className = "column";

        var chartDiv = document.createElement("div");
        chartDiv.classList.add("pass-fail-chart");


        var chartData = new google.visualization.DataTable();

        chartData.addColumn("number", "X");
        chartData.addColumn("number", "Match");
        chartData.addColumn("number", "Mismatch");
        chartData.addRows(data.chartData.lnPassFailData[i]);
        var options = {
            title: "Cummulative L" + (i + 1) + " Success/Failure",
            hAxis: {
                title: "Confidence",
                direction: "-1"
            },
            vAxis: {
                title: "Percentage of Dataset"
            },
            series: {
                0.01: {
                    curveType: "function"
                }
            },
            width: 700,
            height: 400
        };
        var chart = new google.visualization.LineChart(chartDiv);
        chart.draw(chartData, options);
        chartParentDiv.appendChild(chartDiv);
        document.getElementById("passFailGraphs").appendChild(chartParentDiv);
    }
}


function drawHierarchyPassFailGraph() {
    var chartData = new google.visualization.DataTable();
    chartData.addColumn("number", "X");
    chartData.addColumn("number", "L1 Match");
    chartData.addColumn("number", "L1 Mismatch");
    chartData.addColumn("number", "L2 Match");
    chartData.addColumn("number", "L2 Mismatch");
    chartData.addColumn("number", "L3 Match");
    chartData.addColumn("number", "L3 Mismatch");
    chartData.addColumn("number", "L4 Match");
    chartData.addColumn("number", "L4 Mismatch");
    chartData.addColumn("number", "L5 Match");
    chartData.addColumn("number", "L5 Mismatch");
    chartData.addColumn("number", "L6 Match");
    chartData.addColumn("number", "L6 Mismatch");
    chartData.addRows(data.chartData.lnPassFailCombinedData);
    var options = {
        title: "Cummulative Hierarchy Levels Success/Failure",
        hAxis: {
            title: "Confidence",
            direction: "-1"
        },
        vAxis: {
            title: "Percentage of Dataset"
        },
        series: {
            0.01: {
                curveType: "function"
            }
        },
        width: 1400,
        height: 800
    };
    var chart = new google.visualization.LineChart(document.getElementById("Hierarchy_pass_fail_chart"));
    chart.draw(chartData, options);

}


function drawTopKModelScoreGraph() {

    for (var i = 0; i < data.chartData.modelScoreChartData.length; i++) {

        // There is no L6 so do not plot L6 SUccess Failure
        if (i == 5)
            continue;

        var chartParentDiv = document.createElement("div");
        chartParentDiv.className = "column";

        var chartDiv = document.createElement("div");
        chartDiv.classList.add("pass-fail-chart");


        var chartData = new google.visualization.DataTable();

        chartData.addColumn("number", "X");
        chartData.addColumn("number", "Standard");
        chartData.addColumn("number", "Method 1");
        chartData.addColumn("number", "Method 2");
        chartData.addColumn("number", "Method 3");
        chartData.addRows(data.chartData.modelScoreChartData[i]);

        var options = {
            title: "Model Score Top " + (i + 1),
            hAxis: {
                title: "Confidence",
                direction: "-1"
            },
            vAxis: {
                title: "Score Percentage"
            },
            series: {
                0.01: {
                    curveType: "function"
                }
            },
            width: 700,
            height: 400
        };
        var chart = new google.visualization.LineChart(chartDiv);
        chart.draw(chartData, options);
        chartParentDiv.appendChild(chartDiv);
        document.getElementById("score-chart-div").appendChild(chartParentDiv);
    }

    // var data = new google.visualization.DataTable();
    // data.addColumn('number', 'X');
    // data.addColumn('number', 'Standard');
    // data.addColumn('number', 'Method 1');
    // data.addColumn('number', 'Method 2');
    // data.addColumn('number', 'Method 3');
    // data.addRows(
}

function drawMethodScoreChartGraphs() {


    for (var i = 0; i < data.chartData.methodScoreChartData.length; i++) {

        // There is no L6 so do not plot L6 SUccess Failure
        if (i == 5)
            continue;

        var chartParentDiv = document.createElement("div");
        chartParentDiv.className = "column";

        var chartDiv = document.createElement("div");
        chartDiv.classList.add("pass-fail-chart");


        var chartData = new google.visualization.DataTable();

        chartData.addColumn("number", "X");
        chartData.addColumn("number", "Top 1");
        chartData.addColumn("number", "Top 2");
        chartData.addColumn("number", "Top 3");
        chartData.addColumn("number", "Top 4");
        chartData.addColumn("number", "Top 5");
        chartData.addRows(data.chartData.methodScoreChartData[i]);

        var options = {
            title: i > 0 ? "Method " + (i) + " Scoring" : "Standard Scoring Method",
            hAxis: {
                title: "Confidence",
                direction: "-1"
            },
            vAxis: {
                title: "Score Percentage",
            },
            series: {
                0.01: {
                    curveType: "function"
                }
            },
            width: 700,
            height: 400
        };
        var chart = new google.visualization.LineChart(chartDiv);
        chart.draw(chartData, options);
        chartParentDiv.appendChild(chartDiv);
        document.getElementById("score-chart-div").appendChild(chartParentDiv);
    }
}




function addResultHistory() {

    var combinedArrayData = [];


    combinedArrayData.push(["Model", "Match", "Mismatch"]);
    var toDrawModelHistories = [];
    var selectedIndices = [];
    $(".history_check").each(function (index, item) {
        if ($(item).is(":checked")) {
            var id = $(item).data("id");
            selectedIndices.push(parseInt(id));
        }
    });

    modelHistories.forEach(function (model, index) {
        if (selectedIndices.indexOf(index) >= 0) {
            toDrawModelHistories.push(model);
        }
    });

    document.getElementById("Model_Stats_master").innerHTML = "";
    document.getElementById("compare-result-graphs").innerHTML = "";



    toDrawModelHistories.forEach(function (model, index) {

        combinedArrayData.push([model.modelName, model.passCount, model.totalMismatch]);

        var divChild = document.createElement("div");
        divChild.id = "model-history-" + index;
        divChild.className = "column-3";

        document.getElementById("compare-result-graphs").appendChild(divChild);

        google.charts.load("current", {
            "packages": ["bar"]
        });


        google.charts.setOnLoadCallback(function () {

            $(function () {
                var chartData = google.visualization.arrayToDataTable([
                    ["  ", "Match", "Mismatch"],
                    ["Summary", model.passCount, model.totalMismatch]
                ]);
                var options = {
                    title: "Generic Model Overall Result Summary",
                    vAxis: {
                        title: "Images"
                    },
                    width: 400,
                    height: 400
                };
                var chart = new google.charts.Bar(document.getElementById("model-history-" + index));
                chart.draw(chartData, google.charts.Bar.convertOptions(options));
            });

        });

        google.charts.setOnLoadCallback(function () {
            var chartData = google.visualization.arrayToDataTable(combinedArrayData);
            var options = {
                title: "Overall Result Summary",
                vAxis: {
                    title: "Images"
                },
                width: 800,
                height: 600,
                bar: {
                    groupWidth: "30%"
                },
                isStacked: true,
                series: {
                    0: {
                        color: "green"
                    },
                    1: {
                        color: "Indianred"
                    }
                }
            };
            var chart = new google.charts.Bar(document.getElementById("Model_Stats_master"));
            chart.draw(chartData, google.charts.Bar.convertOptions(options));


        });

    });
}

document.addEventListener("DOMContentLoaded", function () {
    $(".history_check").click(function () {
        addResultHistory();

    });
    addResultHistory();


    google.charts.load("current", {
        packages: ["corechart", "bar", "line"]
    });

    google.charts.setOnLoadCallback(drawChart);
    google.charts.setOnLoadCallback(drawResultChart);
    google.charts.setOnLoadCallback(drawTopKResultChart);



    if (data.hasHierarchy) {
        google.charts.setOnLoadCallback(drawPassFailGraph);
        google.charts.setOnLoadCallback(drawLnPassFailGraphs);
        google.charts.setOnLoadCallback(drawHierarchyPassFailGraph);

        google.charts.setOnLoadCallback(drawTopKModelScoreGraph);
        google.charts.setOnLoadCallback(drawMethodScoreChartGraphs);
    }


});