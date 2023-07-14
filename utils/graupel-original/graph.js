/* global fetch, cytoscape, document, window */

fix2 = function (data) {
  var nodes = {};
  var edges = {};
  var labels = {};
  var allEdges = [];
  var newEdges = []

  for (var d in data) {
    var el = data[d];
    if (!el.hasOwnProperty("data")) {

    }
    else if (el.data.hasOwnProperty("source")) {
      edges[el.data.source] = el.data.target;
      allEdges.push([el.data.source, el.data.target]);
    }
    else if (el.classes == "hg-node") {
      nodes[el.data.id] = el;
      // d2.push(data[d]);
    }
    else {
      labels[el.data.id] = el.data.label;
    }
  }

  for (var idx in allEdges) {
    var src = allEdges[idx][0];
    var tmpDst = allEdges[idx][1];
    if (labels.hasOwnProperty(tmpDst)) {
      var label = labels[tmpDst];
      var dst = edges[tmpDst];

      var brakIdx = label.indexOf("(");
      var cls = label.substring(0, brakIdx);

      // var wt = cls == "Sequence" ? 75 : 100;

      var compKey = src + cls;

      // if (!nodes[dst].data.hasOwnProperty(parent)) {

      //   if (!nodes.hasOwnProperty(compKey)) {
      //     var comp = { group: "nodes", data: { id: compKey}, classes: "composite" };
      //     nodes[compKey] = comp;
      //     // var newEdge = { data: { source: src, target: compKey }, classes: cls };
      //     // newEdges.push(newEdge);
      //   }
      //   nodes[dst].data.parent = compKey;

      // }
      // else {
      var newEdge = { data: { source: src, target: dst }, classes: cls };
      newEdges.push(newEdge);
      // }
    }
  }

  var d2 = []
  for (var k in nodes) {
    d2.push(nodes[k])
  }

  for (var k in newEdges) {
    d2.push(newEdges[k])
  }

  // n = document.createTextNode(JSON.stringify(d2));
  // n2 = document.createElement("pre");
  // n2.appendChild(n)
  // document.getElementById('cy').parentElement.appendChild(n2);
  return d2;

};

// makes all hyperedges a regular edge if they have one incomer and one outgoer
collapse_calm_edges = function (selector = "") {
  var nodes = window.cy.$("node.hg-edge");
  nodes = nodes.filter(selector);
  nodes.each(function (edge_node) {
    var incs = edge_node.incomers("edge");
    var outs = edge_node.outgoers("edge");
    if (incs.length == 1 && outs.length == 1) {
      var source = incs[0].source();
      var target = outs[0].target();

      var classes = edge_node.classes();
      var data = edge_node.data();

      window.cy.remove(incs);
      window.cy.remove(outs);
      window.cy.remove(edge_node);

      data.source = source.id();
      data.target = target.id();
      window.cy.add({ data: data, classes: classes });
    }
  });
};

// gives each incidence connected to a hyperedge node all of that hyperedge node's classes (for colouring)
expand_node_edge_classes = function () {
  window.cy.$("node.hg-edge").each(function (edge_node) {
    var node_classes = edge_node.classes();

    var incs = edge_node.incomers("edge");
    var outs = edge_node.outgoers("edge");

    var edges = incs.union(outs);
    edges.each(function (edge) {
      for (i in node_classes) {
        edge.addClass(node_classes[i]);
      }
    });

  });
};

// makes the shaded rectangle that groups nodes
create_compound_nodes = function (cls, label) {
  var nodes = window.cy.$("node." + cls);

  if (nodes.length == 0)
    return;

  var parentID = cls + "-parent";
  window.cy.add({ data: { id: parentID, label: label }, classes: [parentID, "subgraph"] });

  nodes.each(function (node) {
    node.move({ parent: parentID });
  });
}

delete_ordering_edges = function () {
  window.cy.remove(window.cy.$(".ordering"));
};

fix = function () {
  collapse_calm_edges(".cntrl");
  // collapse_calm_edges(".coord");
  expand_node_edge_classes();
  create_compound_nodes("cntrl", "Control Graph");
  create_compound_nodes("coord", "Coordinate Graph");
  group_outgoers("", "cntrl-parent", 6);
  group_linear_sequences(".cntrl");
  // delete_ordering_edges();
  redo_layout();
  autozoom();
  updateDragSubgraphs();
}

mySort = function (a, b) {
  return a.data.label < b.data.label;
};

myLayout = function (subgraph = "") {
  if (subgraph == "coord") {
    return { name: "cola", nodeDimensionsIncludeLabels: true, convergenceThreshold: 0.0001 };

    return {
      // name: 'breadthfirst',
      // name: 'cose',
      // name: 'dagre',
      name: 'cola',
      nodeDimensionsIncludeLabels: true,
      animate: false,
      // concentric: function( node ){ // returns numeric value for each node, placing higher nodes in levels towards the centre
      //   return node.indegree() - node.outdegree();
      //   },
      // elk: { 'algorithm': 'layered'}//, topdownLayout: false, "direction": "RIGHT" },
    };

  }

  if (subgraph == "random") {
    return { name: "random", nodeDimensionsIncludeLabels: true };
  }

  if (subgraph == "circle") {
    return { name: "concentric" };
  }

  if (subgraph == "dagre") {
    return { name: "dagre" };
  }
  if (subgraph == "meta") {
    return {
      name: 'elk',
      animate: false,
      nodeDimensionsIncludeLabels: true,
      elk: { 'algorithm': 'rectpacking', expandNodes: true },
    };
  }

  // layout: {
  //   name: 'dagre',
  //   ranker: 'tight-tree',
  //   sorter: mySort,
  //   nodeDimensionsIncludeLabels: true,
  //   align: "UL"
  //   // ranker: 'longest-path',
  //   // ranker: 'network-simplex',
  //   // cols: 3
  // },
  return {
    name: 'elk',
    nodeDimensionsIncludeLabels: true,
    aspectRatio: 50,
    // elk: {'algorithm': 'mrtree', topdownLayout: false},
    elk: {
      'algorithm': 'layered',
      'aspectRatio': .01,
      'direction': 'RIGHT',
      feedbackEdges: true,
      hierarchyHandling: "INCLUDE_CHILDREN",
      hierarchicalSweepiness: -1,
      thoroughness: 20
    },
  };
  // layout: {name: "cose"},
};

redo_layout = function () {

  var datas = [
    { selector: "", name: "" },
    { selector: ":orphan", name: "meta" },
    // { selector: ".cntrl", name: "coord" },
    // { selector: ":orphan", name: "meta" },
    { selector: ".coord", name: "coord" },
    { selector: ":orphan", name: "meta" },
  ];

  var prevLayout = null;

  datas.forEach(function (d) {
    var nodes = window.cy.$(d.selector);
    nodes = nodes.union(nodes.ancestors());
    var layout = nodes.layout(myLayout(d.name));

    var apply = function () {
      layout.run();
    };

    if (prevLayout == null)
      apply();
    else
      prevLayout.pon("layoutstop").then(apply);

    prevLayout = layout;
  });

};

(function () {
  data = fetch('graph1.json').then(function (res) { return res.json(); });//.then(fix);
  window.cy = cytoscape({
    container: document.getElementById('cy'),

    layout: myLayout(),

    style: fetch('cy-style.json').then(function (res) {
      return res.json();
    }),

    elements: data,
    //    [
    // { data: { label: 'top left' }, classes: 'top-left' },
    // { data: { label: 'top center' }, classes: 'top-center' },
    // { data: { label: 'top right' }, classes: 'top-right' },

    // { data: { label: 'center left' }, classes: 'center-left' },
    // { data: { label: 'center center' }, classes: 'center-center' },
    // { data: { label: 'center right' }, classes: 'center-right' },

    // { data: { label: 'bottom left' }, classes: 'bottom-left' },
    // { data: { label: 'bottom center' }, classes: 'bottom-center' },
    // { data: { label: 'bottom right' }, classes: 'bottom-right' },

    // { data: { label: 'multiline manual\nfoo\nbar\nbaz' }, classes: 'multiline-manual' },
    // { data: { label: 'multiline auto foo bar baz' }, classes: 'multiline-auto' },
    // { data: { label: 'outline' }, classes: 'outline' },

    // { data: { id: 'ar-src' } },
    // { data: { id: 'ar-tgt' } },
    // { data: { source: 'ar-src', target: 'ar-tgt', label: 'autorotate (move my nodes)' }, classes: 'autorotate' },
    // { data: { label: 'background' }, classes: 'background' },

    //     {data: {id: 'ca', label: 'Canada'}},
    //     {data: {id: 'on', label: 'Ontario'}},
    //     {data: {id: 'qc', label: 'Quebec'}},
    //     {data: {source: 'ca', target: 'on', label: 'ON tario'}},
    //     {data: {source: 'ca', target: 'qc'}},
    //   ]
  });

  group_outgoers = function (selector, parentID, minToGroup) {
    var max_out_degree = function (a, b) { return b.outdegree() - a.outdegree(); };

    var up_nodes = window.cy.nodes(selector).sort(max_out_degree);

    var is_orphanish = function (node) {
      if (parentID === null) {
        return node.isOrphan();
      }
      else {
        return node.parent().id() == parentID;
      }
    };

    up_nodes.each(function (up_node) {
      var edges_by_class = {};

      up_node.outgoers("edge").each(function (down_edge) {
        var target = down_edge.target();
        if (!is_orphanish(target))
          return;

        var possible_classes = down_edge.classes();
        var the_class = possible_classes[possible_classes.length - 1];
        if (!edges_by_class.hasOwnProperty(the_class)) {
          edges_by_class[the_class] = []
        }
        edges_by_class[the_class].push(down_edge);
      });

      for (cls in edges_by_class) {
        var edges = edges_by_class[cls];
        if (edges.length < minToGroup) continue;

        var newParentID = up_node.id() + cls;
        var newLabel = up_node.data().label + " " + cls + " (" + newParentID + ")";
        var newParent = { data: { id: newParentID, label: newLabel }, classes: "composite" };
        if (parentID !== null)
          newParent.data.parent = parentID;
        var newEdge = { data: { source: up_node.id(), target: newParentID }, classes: cls };

        window.cy.add([newParent, newEdge]);

        for (i in edges) {
          var edge = edges[i];
          var target = edge.target();

          target.move({ parent: newParentID });

          window.cy.remove(edge);
        }
      }
    });

  };

  groupOutgoers = function () {
    xs = window.cy.nodes().sort(function (a, b) { return b.outgoers().length - a.outgoers().length; });
    xs = xs.toArray();
    for (i in xs) {
      var x = xs[i];
      outs = x.outgoers("edge").toArray();
      byclass = {};

      for (j in outs) {
        var out = outs[j];
        var c = out.classes()[0];
        if (!byclass.hasOwnProperty(c)) {
          byclass[c] = []
        }
        byclass[c].push(out);
      }

      makeEl = function (parID) {
        var label = x.data().label + " " + cls + " (" + parID + ")";
        var el = { data: { id: parID, label: label }, classes: "composite" };
        return window.cy.add(el);
      };

      for (cls in byclass) {
        var parID = x.id() + cls;

        var el = null;

        var edges = byclass[cls];
        if (edges.length > 4) {
          for (k in edges) {
            var edge = edges[k];
            var y = edge.target();

            if (y.isOrphan()) {
              if (el === null) {
                el = makeEl(parID)
              }

              y.move({ parent: parID });

              edge.addClass("hidden");
              // window.cy.remove(edge);
            }
          }

          if (el !== null) {
            newEdge = { data: { source: x.id(), target: el.id() }, classes: cls };
            window.cy.add(newEdge);
          }
        }
        // destID = 
      }

    }
  };

  group_linear_sequences = function (selector) {
    var nodesSeen = new Set();

    window.cy.nodes(selector).each(function (node) {
      if (nodesSeen.has(node.id()))
        return;

      if (node.indegree() > 1 || node.outdegree() > 1)
        return;

      var inc = node.incomers("node");
      while (inc.indegree() == 1) {
        node = inc;
        inc = node.incomers("node");
      }

      var nodesToGroup = [node];

      var out = node.outgoers("node");
      while (out.outdegree() == 1) {
        node = out;
        nodesToGroup.push(node);
        out = node.outgoers("node");
      }

      if(nodesToGroup.length < 2)
        return;

      var parent = node.parent();
      if (parent.classes().indexOf("subgraph") >= 0) {
        parent = window.cy.add({ data: { label: "g", parent: parent.id() }, classes: "composite" });
      }

      nodesToGroup.forEach(function(n){
        nodesSeen.add(n.id());
        n.move({parent: parent.id()});
      });


    });
  }

  // groupLinearSequences = function () {
  //   nodesSeen = []

  //   window.cy.nodes().each(function (node) {
  //     // if(!node.isOrphan()) return;
  //     if (node.children().length > 0) return;
  //     if (node.ancestors().length > 1) return;
  //     if (nodesSeen.indexOf(node.id()) >= 0) return;
  //     var nodesToGroup = [node];
  //     var theParent = null;
  //     if (!node.isOrphan())
  //       theParent = node.parent();

  //     var theNode = node;
  //     while (true) {
  //       var ics = theNode.incomers("node").toArray();
  //       if (ics.length != 1)
  //         break;

  //       theNode = ics[0];
  //       if (!theNode.isOrphan()) {
  //         if (theParent !== null) {
  //           if (theNode.parent() !== theParent)
  //             break;
  //         }
  //         else {
  //           theParent = theNode.parent();
  //         }
  //       }
  //       //   break;

  //       nodesToGroup.push(theNode);
  //     }

  //     theNode = node;
  //     while (true) {
  //       var ics = theNode.outgoers("node").toArray();
  //       if (ics.length != 1)
  //         break;

  //       theNode = ics[0];
  //       // if (!theNode.isOrphan())
  //       //   break;
  //       if (!theNode.isOrphan()) {
  //         if (theParent !== null) {
  //           if (theNode.parent() !== theParent)
  //             break;
  //         }
  //         else {
  //           theParent = theNode.parent();
  //         }
  //       }

  //       nodesToGroup.push(theNode);
  //     }

  //     if (nodesToGroup.length < 2)
  //       return;

  //     newParent = window.cy.add({ label: "seq" });
  //     for (i in nodesToGroup) {
  //       nodesSeen.push(nodesToGroup[i].id());
  //       nodesToGroup[i].move({ parent: newParent.id() });
  //     }
  //     if (theParent !== null)
  //       newParent.move({ parent: theParent.id() });
  //   });

  // };

  groupBodies = function () {
    window.cy.nodes().each(function (node) {
      var bodyEdges = node.outgoers("edge.Body");
      if (bodyEdges.length > 0) {
        var theNodes = bodyEdges.targets();
        theNodes = theNodes.union(theNodes.successors())

        if (theNodes.length > 1) {
          newParent = window.cy.add({ label: "body" });
          theNodes.filter("node").each(function (tm) {
            var toMove = tm;
            while (!toMove.isOrphan() && toMove.id() != newParent.id())
              toMove = toMove.parent();
            if (toMove.id() != newParent.id())
              toMove.move({ parent: newParent.id() });
          });
        }
      }


    });
  }

  doit = function () {

    groupOutgoers();
    // groupLinearSequences();
    // groupBodies();

    l = window.cy.layout(myLayout());
    l.run();


    click = function () {
      var theClass = "target-selected";
      window.cy.$("." + theClass).removeClass(theClass);

      selection = window.cy.$(":selected");
      selection = selection.union(selection.children());
      selection = selection.union(selection.parent());

      selection.openNeighborhood().addClass(theClass);

      selection.each(function(el){
      el.closedNeighborhood().union(el.children()).addClass("target-selected");
      el.predecessors().addClass("target-selected");
      el.successors().addClass("target-selected");
      el.predecessors().ancestors().addClass("target-selected");
      el.children().successors().addClass("target-selected");
      });

      window.cy.pon("tap").then(click);
    };
    window.cy.pon("tap").then(click);
  };

  updateDragSubgraphs = function(){
    var state = document.querySelector('#drag-subgraphs').checked;

    if(state){
      window.cy.$("node.subgraph").unpanify();
    }
    else
    {
      window.cy.$("node.subgraph").panify();
    }
  };

  autozoom = function(){
    window.cy.fit();
  }

  var b = document.querySelector('#fix');
  b.addEventListener('click', fix);
  var b = document.querySelector('#layout');
  b.addEventListener('click', redo_layout);
  var b = document.querySelector('#autozoom');
  b.addEventListener('click', autozoom);
  var b = document.querySelector('#drag-subgraphs');
  b.addEventListener('click', updateDragSubgraphs);
  
  console.log(window.cy.nodes(".coord")); 

  window.cy.ready(doit());

  // doit();
})();