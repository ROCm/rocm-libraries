import { CY_CLASSES } from "../../constants";

/**
 * Creates compound nodes (bounding box) around nodes with specific class
 * @param cy
 * @param className
 * @param parentLabel
 */
export function createCompoundNodes(
  cy: any,
  className: string,
  parentLabel: string
): void {
  let nodes = cy.nodes("." + className);

  if (nodes.length == 0) return;

  let parentID = className + "-parent";
  cy.add({
    data: {
      id: parentID,
      label: parentLabel,
      properties: { composite: true, subgraph: true },
    },
    classes: [parentID, CY_CLASSES.COMPOSITE, CY_CLASSES.SUBGRAPH, className],
  });

  nodes.each(function (node) {
    node.move({ parent: parentID });
  });
}

/**
 * Groups outgoing edges with similar destinations into a single edge and
 * wraps the similar destinations into a composite
 * @param cy
 * @param selector
 * @param parentID
 * @param minToGroup
 */
export function groupOutgoers(
  cy: any,
  selector: string,
  parentID: string,
  minToGroup: number
): void {
  let max_out_degree = function (a, b) {
    return b.outdegree() - a.outdegree();
  };

  let up_nodes = cy.nodes(selector).sort(max_out_degree);

  let is_orphanish = function (node) {
    if (parentID === null) {
      return node.isOrphan();
    } else {
      return node.parent().id() == parentID;
    }
  };

  up_nodes.each(function (up_node) {
    let edges_by_class = {};

    up_node.outgoers("edge").each(function (down_edge) {
      let target = down_edge.target();
      if (!is_orphanish(target)) return;

      let possible_classes = down_edge.classes();
      let the_class = possible_classes[possible_classes.length - 1];
      if (!edges_by_class.hasOwnProperty(the_class)) {
        edges_by_class[the_class] = [];
      }
      edges_by_class[the_class].push(down_edge);
    });

    for (const cls in edges_by_class) {
      let edges = edges_by_class[cls];
      if (edges.length < minToGroup) continue;

      let newParentID = up_node.id() + cls;
      let newLabel =
        up_node.data().label + " " + cls + " (" + newParentID + ")";
      let newParent = {
        data: {
          id: newParentID,
          label: newLabel,
          properties: { composite: true },
        },
        classes: [
          CY_CLASSES.COMPOSITE,
          CY_CLASSES.GROUPED_OUTGOER,
          CY_CLASSES.CONTROL,
        ], // TODO: are they only control graph?
      };
      if (parentID !== null) (<any>newParent.data).parent = parentID;
      let newEdge = {
        data: { source: up_node.id(), target: newParentID },
        classes: cls,
      };

      cy.add([newParent, newEdge]);

      for (const i in edges) {
        let edge = edges[i];
        let target = edge.target();

        target.move({ parent: newParentID });

        cy.remove(edge);
      }
    }
  });
}

/**
 * Makes all hyperedges a regular edge if they have one incomer and one outgoer
 * @param cy
 * @param selector
 */
export function collapseCalmEdges(cy: any, selector: string = ""): void {
  let nodes = cy.nodes(".edge");
  nodes = nodes.filter(selector);
  nodes.each(function (edge_node) {
    let incs = edge_node.incomers("edge");
    let outs = edge_node.outgoers("edge");
    if (incs.length == 1 && outs.length == 1) {
      let source = incs[0].source();
      let target = outs[0].target();

      let classes = edge_node.classes();
      let data = edge_node.data();

      cy.remove(incs);
      cy.remove(outs);
      cy.remove(edge_node);

      data.source = source.id();
      data.target = target.id();
      cy.add({ data: data, classes: classes });
    }
  });
}

/**
 * Gives each edge connected to a hyperedge node all of that hyperedge nodes' classes
 * @param cy
 */
export function expandNodeEdgeClasses(cy: any): void {
  cy.nodes(".edge").each(function (edge_node) {
    let node_classes = edge_node.classes();

    let incs = edge_node.incomers("edge");
    let outs = edge_node.outgoers("edge");

    let edges = incs.union(outs);
    edges.each(function (edge) {
      for (const i in node_classes) {
        edge.addClass(node_classes[i]);
      }
    });
  });
  cy.edges().each((edge) => {
    function expand(classes) {
      if (classes.includes(CY_CLASSES.COORDINATES)) {
        edge.addClass(CY_CLASSES.COORDINATES);
      }
      if (classes.includes(CY_CLASSES.CONTROL)) {
        edge.addClass(CY_CLASSES.CONTROL);
      }
    }
    expand(edge.target().classes());
    expand(edge.source().classes());
  });
}

/**
 * Groups linear sequences / degenerate tree (i.e. node -> node -> ...) into a composite
 * @param cy
 * @param selector
 */
export function groupLinearSequences(cy: any, selector: any): void {
  let nodesSeen = new Set();

  cy.nodes(selector).each(function (node) {
    if (nodesSeen.has(node.id())) return;

    if (node.indegree() > 1 || node.outdegree() > 1) return;

    let inc = node.incomers("node");
    while (inc.indegree() == 1) {
      node = inc;
      inc = node.incomers("node");
    }

    let nodesToGroup = [node];

    let out = node.outgoers("node");
    while (out.outdegree() == 1) {
      node = out;
      nodesToGroup.push(node);
      out = node.outgoers("node");
    }

    if (nodesToGroup.length < 2) return;

    let parent = node.parent();
    if (parent.classes().indexOf(CY_CLASSES.SUBGRAPH) >= 0) {
      parent = cy.add({
        data: {
          label: nodesToGroup[0].data().label,
          parent: parent.id(),
          properties: { composite: true },
        },
        classes: [
          CY_CLASSES.COMPOSITE,
          CY_CLASSES.LINEAR_SEQ,
          selector.substring(1),
        ],
      });
    }

    nodesToGroup.forEach(function (n) {
      nodesSeen.add(n.id());
      n.move({ parent: parent.id() });
    });
  });
}

/**
 * Custom layouts based on subgraphs
 * @param subgraph name
 * @returns cytoscape layout
 */
export function myLayout(subgraph = ""): any {
  if (subgraph == "coord") {
    return {
      name: "cola",
      nodeDimensionsIncludeLabels: true,
      convergenceThreshold: 0.0001,
    };

    return {
      // name: 'breadthfirst',
      // name: 'cose',
      // name: 'dagre',
      name: "cola",
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
      name: "elk",
      animate: false,
      nodeDimensionsIncludeLabels: true,
      elk: { algorithm: "rectpacking", expandNodes: true },
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
    name: "elk",
    nodeDimensionsIncludeLabels: true,
    aspectRatio: 50,
    // elk: {'algorithm': 'mrtree', topdownLayout: false},
    elk: {
      algorithm: "layered",
      aspectRatio: 0.01,
      direction: "RIGHT",
      feedbackEdges: true,
      hierarchyHandling: "INCLUDE_CHILDREN",
      hierarchicalSweepiness: -1,
      thoroughness: 20,
    },
  };
  // layout: {name: "cose"},
}

/**
 * Redoes layout based on node properties or subgraph
 * @param cy
 */
export function redoLayout(cy) {
  let datas = [
    { selector: "", name: "" },
    { selector: ":orphan", name: "meta" },
    // { selector: ".cntrl", name: "coord" },
    // { selector: ":orphan", name: "meta" },
    { selector: ".coord", name: "coord" }, // this look no good
    { selector: ":orphan", name: "meta" },
  ];

  let prevLayout = null;

  datas.forEach(function (d) {
    let nodes = cy.$(d.selector);
    nodes = nodes.union(nodes.ancestors());
    let layout = nodes.layout(myLayout(d.name));

    let apply = function () {
      layout.run();
    };

    if (prevLayout == null) apply();
    else prevLayout.pon("layoutstop").then(apply);

    prevLayout = layout;
  });
}
