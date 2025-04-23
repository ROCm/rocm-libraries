/* eslint-disable eqeqeq */
/*global data, labelSummary, imageSummary, $, modelHistories, hierarchyData*/

// eslint-disable-next-line no-unused-vars
function openNav() {
    document.getElementById("mySidenav").style.width = "250px";
    document.getElementById("main").style.marginLeft = "250px";
}

// eslint-disable-next-line no-unused-vars
function closeNav() {
    document.getElementById("mySidenav").style.width = "0";
    document.getElementById("main").style.marginLeft = "0";
}

// eslint-disable-next-line no-unused-vars
function myreload() {
    location.reload();
}



function insertIntoTopKTable() {
    var table = document.getElementById("top-5-match-table");
    var row = table.insertRow(-1);
    var row2 = table.insertRow(-1);

    data.topKStats.forEach(function (element) {
        var cell = row.insertCell(-1);
        cell.innerHTML = "<b>" + element.matches + "</b>";
        var cell2 = row2.insertCell(-1);
        cell2.innerHTML = "<b>" + element.accuracyPer.toFixed(2) + " %</b>";
    });
}

function showStatsOnTable() {
    $("b[id ^= 'stat']").each(function () {
        var property = this.id.split("_")[1];
        var round = Number(this.getAttribute("data-round"));
        if (data.stats[property])
            this.innerHTML = data.stats[property].toFixed(round);

    });

    $("#modelNameText").html(data.stats.modelName + " Overall Summary");
}

function loadHierarchySummary() {
    var myTableBody = $("#hierarchy-summary-table").find("tbody");
    myTableBody.empty();
    var f = 0.99;
    var topKPassFail = hierarchyData.topKPassFail;
    var topKHierarchyPassFail = hierarchyData.topKHierarchyPassFail;

    for (var i = 99; i >= 0; i--) {

        var row = $("<tr>");
        row.append($("<td>").attr("class", "blue").text(f.toFixed(2)));
        row.append($("<td>").attr("class", "blue").text(topKPassFail[i][0]));
        row.append($("<td>").attr("class", "blue").text(topKPassFail[i][1]));

        for (var j = 0; j < 12; j++) {
            row.append($("<td>").attr("class", "blue").text(topKHierarchyPassFail[i][j]));
        }
        myTableBody.append(row);
        f = f - 0.01;
    }
}


function loadLabelSummary(labelSummaryLocal) {
    var myTableBody = $("#label-summary-table").find("tbody");
    myTableBody.empty();
    labelSummaryLocal.forEach(function (label) {
        var row = $("<tr>");
        var id = label.id;
        var totalImageClass = label.totalImages > 0 ? "color-green" : "";

        var misclassifiedClass = label.misclassifiedTop1 === 0 ? "color-green" : (label.totalImages > 0 ? "color-red" : "color-black");
        row.append($("<td>").attr({
            "class": "blue filter-image",
            "data-id": label.id
        }).text(label.id));

        row.append($("<td>").attr({
            "class": "blue left-align filter-image",
            "data-id": label.id
        }).text(label.label));

        row.append($("<td>").attr("class", "blue " + totalImageClass).text(label.totalImages));
        row.append($("<td>").attr("class", "blue").text(label.matchedTop1Per.toFixed(2)));
        row.append($("<td>").attr("class", "blue").text(label.matchedTop5Per.toFixed(2)));
        row.append($("<td>").attr("class", "blue").text(label.match1));
        row.append($("<td>").attr("class", "blue").text(label.match2));
        row.append($("<td>").attr("class", "blue").text(label.match3));
        row.append($("<td>").attr("class", "blue").text(label.match4));
        row.append($("<td>").attr("class", "blue").text(label.match4));
        row.append($("<td>").attr({
            "class": "blue filter-image " + misclassifiedClass,
            "data-id": label.id,
            "data-type": "misclassified"
        }).text(label.misclassifiedTop1));

        row.append($("<td>").html("<input id=\"id_\"" + id + " name=\"id[" + id + "]\" type=\"checkbox\" value=\"" + id + "\" onClick=\"highlightRow(this);\"></input>"));
        myTableBody.append(row);
    });


}

// eslint-disable-next-line no-unused-vars
function filterLabelTable(rowNum, Datavar) {
    var labelSummaryFiltered = [];
    var allEmpty = true;
    // eslint-disable-next-line no-unused-vars
    $("input[id ^= \"fl_\"]").each(function (element) {
        var property = this.id.split("_")[1];
        var value = this.value;
        var compare = this.getAttribute("data-compare");
        if (value) {
            allEmpty = false;

            labelSummary.forEach(function (label) {
                var compareResult = false;

                if (compare === "contains") {
                    compareResult = label[property].toLowerCase().includes(value.toLowerCase());

                } else {
                    compareResult = label[property] == value;
                }
                if (compareResult) {
                    labelSummaryFiltered.push(label);
                }
            });
        }
    });
    if (allEmpty) {
        loadLabelSummary(labelSummary);
    } else {
        loadLabelSummary(labelSummaryFiltered);
    }
}

// eslint-disable-next-line no-unused-vars
function clearLabelFilter() {
    $("input[id ^= \"fl_\"]").val("");

    loadLabelSummary(labelSummary);
}

// eslint-disable-next-line no-unused-vars
function highlightRow(element) {
    var parentRow = element.parentElement.parentElement;
    parentRow.classList.toggle("highlight-row");

}


//Create copy of imageSummary
var imageSummaryFiltered = JSON.parse(JSON.stringify(imageSummary));

// eslint-disable-next-line no-unused-vars
function clearResultFilter() {
    $("input[id ^= \"fli_\"]").val("");
    imageSummaryFiltered = JSON.parse(JSON.stringify(imageSummary));
    loadImageResults(imageSummaryFiltered);
    updateItems();
}

function filterResultTable(e) {
    imageSummaryFiltered = [];
    var allEmpty = true;
    if (e) {

        // eslint-disable-next-line no-unused-vars
        var code = (e.keyCode ? e.keyCode : e.which);
    }
    var combinationOp = $("input:radio[name='filterType']:checked").val();
    var notOp = $("#not-op").is(":checked");


    var filters = {};

    // eslint-disable-next-line no-unused-vars
    $("input[id ^= \"fli_\"]").each(function (element) {
        var allNames = this.id.split("_");
        var property_index = null;
        var prop_name_new = allNames[1];
        if (allNames[2]) {
            property_index = parseInt(allNames[2], 10);
            prop_name_new = prop_name_new + "_" + property_index;
        }
        var value = this.value;
        var op = $("#fli_op_" + prop_name_new).val();
        if (value) {
            allEmpty = false;
            filters[this.id] = [];
            filters[this.id].push(op);
            filters[this.id].push(value);

        }
    });

    imageSummaryFiltered = imageSummary.filter(function (item) {
        var allTrue = true;
        var anyTrue = false;

        for (var key in filters) {
            var allNames = key.split("_");
            var property = allNames[1];
            var property_index = null;
            if (allNames[2]) {
                property_index = parseInt(allNames[2], 10);
            }
            var itemValue = item[property];
            if (allNames[2] >= 0) {
                itemValue = itemValue[property_index];
            }

            if (!compare(itemValue, filters[key][1], filters[key][0])) {
                allTrue = false;
            } else {
                anyTrue = true;
            }


        }
        if (notOp === true) {
            anyTrue = !anyTrue;
            allTrue = !allTrue;
        }
        if (combinationOp == "or") {
            return anyTrue;
        } else if (combinationOp == "and") {
            return allTrue;
        }

        return true;

    });

    if (allEmpty) {
        var perPage = 50;
        var pageNumber = 1;
        imageSummaryFiltered = JSON.parse(JSON.stringify(imageSummary));
        var showFrom = perPage * (pageNumber - 1);
        var showTo = showFrom + perPage;
        loadImageResults(imageSummaryFiltered.slice(showFrom, showTo));

    } else {
        loadImageResults(imageSummaryFiltered);
    }
    updateItems();
}



function loadImageResults(imageSummaryLocal) {
    var myTableBody = $("#image-results-table").find("tbody");
    myTableBody.empty();
    var count = 0;
    imageSummaryLocal.forEach(function (im) {

        if (count > 40)
            return;

        count++;

        var row = $("<tr>");

        // var id = im.imgName;

        var matchClass = im.match > 0 ? "color-green" : "color-red";
        // var imgLink = $('a').attr('href', im.filePath).attr('target', '_blank').text(im.imageName);

        var imgHtml = "<img onclick=\"showModal(event)\" src=\"" +
            im.filePath +
            "\" data-alt=\"<b>GROUND TRUTH:</b>" +
            im.gtText + "<br><b>CLASSIFIED AS:</b>" +
            im.labelTexts[0] + "\" width=\"30\" height=\"30\">";
        var imgLinkHtml = "<a href=\"" + im.filePath + "\" target=\"_blank\">" + im.imageName + "</a>";

        row.append($("<td>").attr("class", "imgcell").html(imgHtml));
        row.append($("<td>").html(imgLinkHtml));
        row.append($("<td>").attr("class", "imgcell left-align").text(im.gtText));
        row.append($("<td>").attr("class", "imgcell").text(im.gt));

        for (var i = 0; i < 5; i++) {
            row.append($("<td>").attr("class", "imgcell").text(im.labels[i]));
        }
        row.append($("<td>").attr("class", "imgcell bold " + matchClass).text(im.match));

        for (i = 0; i < 5; i++) {
            row.append($("<td>").attr("class", "imgcell left-align").text(im.labelTexts[i]));
        }

        for (i = 0; i < 5; i++) {
            row.append($("<td>").attr("class", "imgcell").text(im.probs[i].toFixed(4)));
        }
        myTableBody.append(row);
    });


}

// eslint-disable-next-line no-unused-vars
function showModal(event) {
    var modal = document.getElementById("myModal");
    var modalImg = document.getElementById("img01");
    var modalCaption = document.getElementById("caption");
    modal.style.display = "block";
    modalCaption.innerHTML = event.target.dataset.alt;
    modalImg.src = event.target.src;
}

// eslint-disable-next-line no-unused-vars
function closeModal() {
    var modal = document.getElementById("myModal");
    modal.style.display = "none";

}

//Update pagnator -- needs to be called when items change dynamically
function updateItems() {
    var $paginator = $(".pagination-page");
    var items = imageSummaryFiltered;
    $paginator.pagination("updateItems", items.length);
    // Next, we'll re-select the page so the `onPageClick` function we already
    // programmed will handle hiding and showing the correct content for us.
    // Note that the page we were on before may no longer exist
    // (i.e. if elements were removed and that page is no longer required),
    // so we will make sure to select an existing page.
    var page = Math.min(
        $paginator.pagination("getCurrentPage"),
        $paginator.pagination("getPagesCount")
    );

    $("#table-item-count").html("Total Items: " + items.length);
    $paginator.pagination("selectPage", page);
}


jQuery(function ($) {
    var items = imageSummaryFiltered;

    var numItems = items.length;
    var perPage = 50;
    // Only show the first 50 (or first `per_page`) items initially.
    // items.slice(perPage).hide();
    // Now setup the pagination using the `.pagination-page` div.
    $(".pagination-page").pagination({
        items: numItems,
        itemsOnPage: perPage,
        cssStyle: "light-theme",
        // This is the actual page changing functionality.
        onPageClick: function (pageNumber) {
            // We need to show and hide `tr`s appropriately.
            var showFrom = perPage * (pageNumber - 1);
            var showTo = showFrom + perPage;
            loadImageResults(imageSummaryFiltered.slice(showFrom, showTo));

        }
    });




    // We'll call this now so the initial items can be loaded.
    updateItems();

    // EDIT: Let's cover URL fragments (i.e. #page-3 in the URL).
    // More thoroughly explained (including the regular expression) in: 
    // https://github.com/bilalakil/bin/tree/master/simplepagination/page-fragment/index.html
    // We'll create a function to check the URL fragment
    // and trigger a change of page accordingly.
    function checkFragment() {
        // If there's no hash, treat it like page 1.
        var hash = window.location.hash || "#page-1";
        // We'll use a regular expression to check the hash string.
        hash = hash.match(/^#page-(\d+)$/);
        if (hash) {
            // The `selectPage` function is described in the documentation.
            // We've captured the page number in a regex group: `(\d+)`.
            $(".pagination-page").pagination("selectPage", parseInt(hash[1], 10));
        }
    }

    // We'll call this function whenever back/forward is pressed...
    $(window).bind("popstate", checkFragment);
    // ... and we'll also call it when the page has loaded
    // (which is right now).
    checkFragment();
});





function loadScorings() {
    var myTableBody = $("#standard-scoring-table").find("tbody");
    myTableBody.empty();
    var row = $("<tr>");
    data.scores.matchCounts.forEach(function (value) {
        row.append($("<td>").text(value));
    });
    myTableBody.append(row);

    row = $("<tr>");
    data.scores.modelScores.forEach(function (value) {
        row.append($("<td>").text(value.toFixed(2) + "%"));
    });
    myTableBody.append(row);


    if (data.hasHierarchy) {
        var tableIds = ["#method1-scoring-table", "#method2-scoring-table", "#method3-scoring-table"];
        var scores = [data.scores.method1Scores, data.scores.method2Scores, data.scores.method3Scores];

        tableIds.forEach(function (tableId, i) {
            myTableBody = $(tableId).find("tbody");
            myTableBody.empty();
            var row = $("<tr>");

            data.scores.matchCounts.forEach(function (value) {
                row.append($("<td>").text(value));
            });
            myTableBody.append(row);

            row = $("<tr>");
            scores[i].forEach(function (value) {
                row.append($("<td>").text(value.toFixed(2) + "%"));
            });
            myTableBody.append(row);


        });
        $("#hierarchial-scoring").toggleClass("hidden");
    }


}


function compare(a, b, operator) {

    switch (operator) {
        case "contains":
            return a.toString().toLowerCase().includes(b.toString().toLowerCase());
        case "eq":
            return a == b;
        case "neq":
            return a != b;
        case "gt":
            return a > b;
        case "lt":
            return a < b;
        case "gte":
            return a >= b;
        case "lte":
            return a <= b;
        default:
            return a == b;
    }
}

function sortImageResults(fieldName, direction) {
    var fieldNameArr = fieldName.split("_");
    var propertyName = fieldNameArr[0];
    var propertyIndex;
    if (fieldNameArr[1]) {
        propertyIndex = parseInt(fieldNameArr[1], 10);
    }
    imageSummaryFiltered.sort(function (a, b) {
        var aVal = a[propertyName];
        var bVal = b[propertyName];
        if (fieldNameArr[1]) {
            aVal = aVal[propertyIndex];
            bVal = bVal[propertyIndex];
        }
        var result = (aVal < bVal) ? 1 : -1;
        if (direction == -1) {
            result = result * -1;
        }
        return result;
    });
    loadImageResults(imageSummaryFiltered);
}


function addToOldDataList() {
    modelHistories.forEach(function (model, index) {
        var tbody = document.getElementById("history-tbody");

        var row = document.createElement("tr");

        var col = document.createElement("td");
        var label = document.createElement("label");
        label.classList.add("result-history-check");
        var elem = document.createElement("input");
        elem.type = "checkbox";
        elem.classList.add("history_check");
        elem.dataset.id = index;
        elem.id = "history_" + index;
        if (index > modelHistories.length - 5)
            elem.checked = true;
        // elem.classList.add('result-history-check');
        label.appendChild(elem);
        // document.createTextNode(model.modelName + " | " + model.genDate);
        col.append(label);
        row.append(col);
        col = document.createElement("td");
        col.append(document.createTextNode(model.modelName));
        row.appendChild(col);
        col = document.createElement("td");
        col.append(document.createTextNode(model.genDate));
        row.appendChild(col);
        col = document.createElement("td");
        col.append(document.createTextNode(model.totalImages));
        row.appendChild(col);
        tbody.appendChild(row);

    });

    //Shortcut to use table sort to filter by date
    setTimeout(function () {
        document.getElementById("date_sort_header").click();
    }, 1000);
    setTimeout(function () {
        document.getElementById("date_sort_header").click();
    }, 2000);


}

document.addEventListener("DOMContentLoaded", function () {
    $("#generation-date").html(data.summaryGenerationDate);

    insertIntoTopKTable();
    showStatsOnTable();
    loadLabelSummary(labelSummary);
    loadScorings();

    addToOldDataList();


    if (data.hasHierarchy) {
        $(".has-hierarchy").toggleClass("hidden");
        loadHierarchySummary();
    }

    $(".filter-image").on("click", function (event) {
        var element = event.target;
        var targetValue = element.dataset.id;
        var toMisclassified = element.dataset.type;
        $("input[id ^= \"fli_\"]").val("");

        if (!toMisclassified) {
            $("#fli_gt").val(targetValue);
        } else {
            $("#fli_labels_0").val(targetValue);
            $("#fli_match").val(1);
            $("#fli_op_match").val("neq");
        }
        filterResultTable(event);
        window.location.href = "#table4";

    });

    $("#not-op").click(function () {
        filterResultTable();
    });
    $("input:radio[name='filterType']").click(function () {
        filterResultTable();
    });
    // loadImageResults(imageSummary);

    $("select[id ^= \"fli_op_\"]").change(function () {
        filterResultTable();
    });


    // eslint-disable-next-line no-unused-vars
    $(".sort-field").click(function (event) {
        var targetField = $(this).data("field");
        var direction = $(this).data("sort-dir");

        $(".sort-field span").html("");

        direction = direction ? parseInt(direction, 10) : 1;
        sortImageResults(targetField, direction);
        direction = direction * -1;
        $(this).data("sort-dir", direction);
        var $this = $(this).find("span");
        $this.html(direction == -1 ? "&nbsp;&#x25B4;" : "&nbsp;&#x25BE;");

    });
});