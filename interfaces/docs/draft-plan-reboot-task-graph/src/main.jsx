import React, { StrictMode, useCallback, useEffect, useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  Background,
  ControlButton,
  Controls,
  Handle,
  MarkerType,
  MiniMap,
  Position,
  ReactFlow,
  ReactFlowProvider,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import './styles.css';
import { taskEdges, taskNodes } from './graph-data';
import { glossaryAliases, glossaryEntries } from './glossary-data';
import { specificationByHref, specificationById, specificationEntries } from './specification-data';

const glossaryById = Object.fromEntries(glossaryEntries.map((entry) => [entry.id,entry]));
const testCatalog = taskNodes.flatMap((task) => (task.tests ?? [])
  .filter((item) => item.testId)
  .map((test) => {
    const ownerTask=taskNodes.find((candidate) => candidate.id === test.ownerTask) ?? task;
    return {...test,definedByTaskId:task.id,taskId:ownerTask.id,taskLabel:ownerTask.label};
  }));
const testById = Object.fromEntries(testCatalog.map((test) => [test.testId,test]));
const taskAliases = {'c boundary':'i10'};
const escapePattern = (value) => value.replace(/[.*+?^${}()|[\]\\]/g,'\\$&');
const linkedTermPattern = [...Object.keys(glossaryAliases),...Object.keys(taskAliases)]
  .sort((left,right) => right.length-left.length)
  .map(escapePattern)
  .join('|');
const richTextPattern = new RegExp(`(\\btest-\\d+\\b|\\bi\\d{2}\\b|\\b(?:${linkedTermPattern})\\b)`,'gi');

const positions = {
  'i08':[60,460], 'i10':[340,460],
  'i11':[640,250], 'i12':[640,460], 'i13':[640,670],
  'i14':[1060,250], 'i15':[1060,460], 'i16':[1060,670], 'i17':[1370,460],
  'phase-1-complete':[1650,460],
  'i18':[360,368], 'i19':[680,80], 'i20':[680,215], 'i22':[680,350],
  'i23':[680,485], 'i24':[680,620], 'i21':[1000,148], 'i25':[1320,350],
  'i26':[360,840], 'i27':[680,840], 'i28':[1000,840], 'i29':[1600,840],
  'i30':[360,1080], 'i31':[680,1080], 'i32':[1000,1080],
  'phase-2-complete':[1900,840],
  'i33':[380,120], 'i34':[700,120], 'i35':[1020,120], 'i36':[1340,120],
  'i37':[380,363], 'i38':[700,363], 'i39':[1020,363],
  'i40':[1660,309], 'i41':[1660,444],
  'i42':[380,660], 'i43':[700,660], 'i44':[1020,588], 'i45':[1020,732],
  'i46':[1340,660],
  'i47':[380,948], 'i48':[700,948], 'i49':[1020,948],
  'i50':[2050,660],
};

const slides = [
  {
    id:'overview',
    title:'Delivery overview',
    description:'Three delivery phases organized around macro deliverables and independently reviewable pull requests.',
    includes:() => false,
  },
  {
    id:'phase-1',
    title:'Phase 1 · hipBLASLt MVP',
    description:'Prove the representative GEMM API/ABI subset through its mini-app, facade, provider, and qualification.',
    includes:(item) => item.phase === 'Phase 1' || item.id === 'phase-1-complete',
  },
  {
    id:'phase-2',
    title:'Phase 2 · parallel expansion',
    description:'Scale horizontally across hipBLASLt, vertically to PyTorch and MIOpen, and onboard the external JIT path.',
    includes:(item) => item.phase.startsWith('Phase 2') || item.id === 'phase-2-complete',
  },
  {
    id:'phase-3',
    title:'Phase 3 · named component expansions',
    description:'Sparse, conditional hipSPARSELt, optional solver/RAND providers, FFT, and tensor — each with explicit bounded nodes.',
    includes:(item) => item.phase.startsWith('Phase 3'),
  },
];

const phaseOverviewNodes = [
  {
    id:'overview-phase-1',
    position:{x:60,y:250},
    data:{
      phase:'Phase 1',
      title:'hipBLASLt facade path',
      counts:'9 PRs · 2 deliverable groups',
      phaseIndex:1,
      macros:[
        {title:'Facade MVP',description:'Provider boundary, broker, provider, and facade.'},
        {title:'Integration and compatibility',description:'Shadow deployment, backend boundary, ABI coexistence, and qualification.'},
      ],
    },
  },
  {
    id:'overview-phase-2',
    position:{x:570,y:190},
    data:{
      phase:'Phase 2',
      title:'Parallel expansion',
      counts:'15 PRs · 3 deliverable groups',
      phaseIndex:2,
      macros:[
        {title:'Full hipBLASLt surface',description:'Complete public C API and ABI coverage.'},
        {title:'PyTorch dependency path',description:'rocBLAS, hipBLAS, MIOpen, and PyTorch qualification.'},
        {title:'JIT path',description:'Provider, facade, and shadow qualification.'},
      ],
    },
  },
  {
    id:'overview-phase-3',
    position:{x:1080,y:100},
    data:{
      phase:'Phase 3',
      title:'Component-family expansion',
      counts:'14 PRs · 4 spikes · 5 deliverable groups',
      phaseIndex:3,
      macros:[
        {title:'Sparse family',description:'rocSPARSE, hipSPARSE, and optional Sparse-Lt.'},
        {title:'Alternate providers',description:'rocSOLVER and rocRAND alternatives.'},
        {title:'FFT family',description:'rocFFT, hipFFT, and hipFFTW.'},
        {title:'Tensor family',description:'hipTensor facade path.'},
        {title:'Integration',description:'Phase 3 release-candidate gate.'},
      ],
    },
  },
];

const statusLabel = { ready:'Ready', blocked:'', later:'Later phase' };

function TaskNode({ data, selected }) {
  return (
    <article className={`task-card ${data.status} ${data.type.toLowerCase()} ${selected ? 'selected' : ''}`}>
      <Handle id="target-left" type="target" position={Position.Left} className="task-handle" />
      <Handle id="target-right" type="target" position={Position.Right} className="task-handle" />
      <Handle id="target-top" type="target" position={Position.Top} className="task-handle" />
      <Handle id="target-bottom" type="target" position={Position.Bottom} className="task-handle" />
      <div className="task-topline">
        {!data.navigation && !data.milestone && <span className="task-id">{data.id}</span>}
        {!data.milestone && <span className="status-dot" title={statusLabel[data.status] || undefined} />}
      </div>
      <h3>{data.label}</h3>
      <Handle id="source-left" type="source" position={Position.Left} className="task-handle" />
      <Handle id="source-right" type="source" position={Position.Right} className="task-handle" />
      <Handle id="source-top" type="source" position={Position.Top} className="task-handle" />
      <Handle id="source-bottom" type="source" position={Position.Bottom} className="task-handle" />
    </article>
  );
}

function WorkGroupNode({ data }) {
  return (
    <section className={`work-group ${data.optional ? 'optional' : ''}`}>
      {data.connectable && <Handle id="target-left" type="target" position={Position.Left} className="group-handle" />}
      <div className="work-group-title">{data.label}</div>
      {data.connectable && <Handle id="source-right" type="source" position={Position.Right} className="group-handle" />}
    </section>
  );
}

function PhaseOverviewNode({ data }) {
  return (
    <article className="phase-overview-card">
      <Handle id="target-left" type="target" position={Position.Left} className="overview-handle" />
      <div className="phase-overview-kicker">{data.phase}</div>
      <h3>{data.title}</h3>
      <p className="phase-overview-counts">{data.counts}</p>
      <div className="phase-overview-macros">
        {data.macros.map((macro) => <div className="phase-overview-macro" key={macro.title}>
          <strong>{macro.title}</strong>
          <span>{macro.description}</span>
        </div>)}
      </div>
      <p className="phase-overview-open">Click to open detailed tasks</p>
      <Handle id="source-right" type="source" position={Position.Right} className="overview-handle" />
    </article>
  );
}

const nodeTypes = { task:TaskNode, workGroup:WorkGroupNode, phaseOverview:PhaseOverviewNode };

function App() {
  const query = new URLSearchParams(window.location.search);
  const requestedPhase = Number.parseInt(query.get('phase') ?? '', 10);
  const requestedNode = query.get('node');
  const requestedTask=taskNodes.find((item) => item.id === requestedNode);
  const requestedTaskSlide=requestedTask ? slides.findIndex((candidate) => candidate.includes(requestedTask)) : -1;
  const requestedPhaseSlide=Number.isFinite(requestedPhase) ? slides.findIndex((candidate) => candidate.id === `phase-${requestedPhase}`) : -1;
  const initialSlide=requestedTaskSlide >= 0 ? requestedTaskSlide : requestedPhaseSlide >= 0 ? requestedPhaseSlide : 0;
  const [selectedId, setSelectedId] = useState(taskNodes.some((item) => item.id === requestedNode) ? requestedNode : null);
  const [activeSlide, setActiveSlide] = useState(initialSlide);
  const [slideDirection, setSlideDirection] = useState('next');
  const [helpOpen, setHelpOpen] = useState(false);
  const [glossaryId, setGlossaryId] = useState('public-api');
  const [testCatalogOpen, setTestCatalogOpen] = useState(false);
  const [catalogTestId, setCatalogTestId] = useState('test-1');
  const [specCatalogOpen, setSpecCatalogOpen] = useState(false);
  const [catalogSpecificationId, setCatalogSpecificationId] = useState('architecture-component-model');
  const [nodePositions, setNodePositions] = useState(() => {
    try {
      const layoutKey='mathlibs-task-graph-layout-v14';
      const savedV14=localStorage.getItem(layoutKey);
      const saved = JSON.parse(savedV14 ?? localStorage.getItem('mathlibs-task-graph-layout-v13') ?? localStorage.getItem('mathlibs-task-graph-layout-v12') ?? localStorage.getItem('mathlibs-task-graph-layout-v11') ?? '{}');
      const merged={...positions,...saved};
      taskNodes.filter((item) => item.phase === 'Phase 1' || item.id === 'phase-1-complete')
        .forEach((item) => { merged[item.id]=positions[item.id]; });
      if (!savedV14) {
        ['i26','i27','i28','i29','i30','i31','i32','phase-2-complete']
          .forEach((id) => { merged[id]=positions[id]; });
        ['i40','i41','i50'].forEach((id) => { merged[id]=positions[id]; });
      }
      return merged;
    } catch {
      return {...positions};
    }
  });
  const nodeById = useMemo(() => Object.fromEntries(taskNodes.map((item) => [item.id,item])), []);
  const dependencies = useMemo(() => {
    const parents={}, children={};
    taskNodes.forEach((item) => { parents[item.id]=[]; children[item.id]=[]; });
    taskEdges.forEach(([from,to]) => { parents[to].push(from); children[from].push(to); });
    return {parents,children};
  }, []);

  useEffect(() => {
    localStorage.setItem('mathlibs-task-graph-layout-v14',JSON.stringify(nodePositions));
  }, [nodePositions]);

  useEffect(() => {
    if (!helpOpen && !testCatalogOpen && !specCatalogOpen) return undefined;
    const closeOnEscape = (event) => {
      if (event.key === 'Escape') {
        setHelpOpen(false);
        setTestCatalogOpen(false);
        setSpecCatalogOpen(false);
      }
    };
    window.addEventListener('keydown',closeOnEscape);
    return () => window.removeEventListener('keydown',closeOnEscape);
  }, [helpOpen,testCatalogOpen,specCatalogOpen]);

  const connected = useMemo(() => {
    if (!selectedId) return new Set();
    const result=new Set([selectedId]);
    const walk=(id,map) => map[id].forEach((next) => { if(!result.has(next)){ result.add(next); walk(next,map); } });
    walk(selectedId,dependencies.parents);
    walk(selectedId,dependencies.children);
    return result;
  }, [selectedId,dependencies]);

  const slide = slides[activeSlide];
  const visibleIds = useMemo(
    () => new Set(taskNodes.filter(slide.includes).map((item) => item.id)),
    [slide],
  );

  const nodes = useMemo(() => {
    if (slide.id === 'overview') {
      return phaseOverviewNodes.map((node) => ({
        id:node.id,
        type:'phaseOverview',
        position:node.position,
        style:{width:440},
        draggable:false,
        selectable:true,
        data:node.data,
      }));
    }
    const taskFlowNodes=taskNodes.filter(slide.includes).map((item) => {
    const isPhaseTwoOrigin = item.id === 'phase-1-complete' && slide.id === 'phase-2';
    const isPhaseThreeOrigin = item.id === 'phase-2-complete' && slide.id === 'phase-3';
    let navigationLabel=item.label;
    if (isPhaseTwoOrigin) navigationLabel='Phase 1';
    if (isPhaseThreeOrigin) navigationLabel='Phase 2';
    return {
      id:item.id,
      type:'task',
      position:isPhaseTwoOrigin
        ? {x:40,y:638}
        : isPhaseThreeOrigin
          ? {x:40,y:597}
          : {x:nodePositions[item.id][0],y:nodePositions[item.id][1]},
      draggable:item.phase !== 'Phase 1' && !item.navigation,
      selectable:!item.milestone,
      data:{...item,label:navigationLabel},
      className:[
        item.phase === 'Phase 1' ? 'phase-one-node' : '',
        selectedId && !connected.has(item.id) ? 'faded' : '',
      ].filter(Boolean).join(' '),
    };
    });
    const makeGroup = (id,label,members,options={}) => {
      const paddingX=options.paddingX ?? 30;
      const paddingTop=options.paddingTop ?? 45;
      const paddingBottom=options.paddingBottom ?? 45;
      const width=250;
      const height=104;
      const points=members.map((member) => nodePositions[member]);
      const minX=Math.min(...points.map(([x]) => x));
      const minY=Math.min(...points.map(([,y]) => y));
      const maxX=Math.max(...points.map(([x]) => x+width));
      const maxY=Math.max(...points.map(([,y]) => y+height));
      return {
        id,
        type:'workGroup',
        position:{x:minX-paddingX,y:minY-paddingTop},
        style:{width:maxX-minX+paddingX*2,height:maxY-minY+paddingTop+paddingBottom},
        draggable:false,
        selectable:false,
        data:{label,optional:options.optional ?? false,connectable:options.connectable ?? false},
        zIndex:options.zIndex ?? -1,
      };
    };
    if (slide.id === 'phase-2') {
      return [
        makeGroup('phase-2-horizontal-group','Full hipBLASLt surface',['i18','i19','i20','i21','i22','i23','i24','i25']),
        makeGroup('phase-2-pytorch-group','PyTorch dependency path',['i26','i27','i28','i29']),
        makeGroup('phase-2-jit-group','JIT path',['i30','i31','i32']),
        ...taskFlowNodes,
      ];
    }
    if (slide.id === 'phase-3') {
      return [
        makeGroup('phase-3-sparse-group','Sparse family',['i33','i34','i35','i36'],{connectable:true}),
        makeGroup('phase-3-sparse-lt-group','Optional Sparse-Lt',['i37','i38','i39'],{optional:true,connectable:true,zIndex:-1}),
        makeGroup('phase-3-alternate-group','Alternate providers',['i40','i41'],{connectable:true}),
        makeGroup('phase-3-fft-group','FFT family',['i42','i43','i44','i45','i46'],{connectable:true}),
        makeGroup('phase-3-tensor-group','Tensor family',['i47','i48','i49'],{connectable:true}),
        ...taskFlowNodes,
      ];
    }
    if (slide.id !== 'phase-1') return taskFlowNodes;
    return [
      {
        id:'phase-1-core-group',
        type:'workGroup',
        position:{x:300,y:190},
        style:{width:605,height:590},
        draggable:false,
        selectable:false,
        data:{label:'Facade MVP',connectable:true},
        zIndex:-1,
      },
      {
        id:'phase-1-integration-group',
        type:'workGroup',
        position:{x:1020,y:190},
        style:{width:305,height:590},
        draggable:false,
        selectable:false,
        data:{label:'Integration and compatibility',connectable:true},
        zIndex:-1,
      },
      ...taskFlowNodes,
    ];
  }, [selectedId,connected,slide,nodePositions]);

  const edgeHandles = useCallback((source,target) => {
    if (source === 'phase-1-complete' || source === 'phase-2-complete') {
      return {sourceHandle:'source-right',targetHandle:'target-left'};
    }
    const [sourceX,sourceY] = nodePositions[source];
    const [targetX,targetY] = nodePositions[target];
    const deltaX = targetX-sourceX;
    const deltaY = targetY-sourceY;
    if (Math.abs(deltaX) >= Math.abs(deltaY)) {
      return deltaX >= 0
        ? {sourceHandle:'source-right',targetHandle:'target-left'}
        : {sourceHandle:'source-left',targetHandle:'target-right'};
    }
    return deltaY >= 0
      ? {sourceHandle:'source-bottom',targetHandle:'target-top'}
      : {sourceHandle:'source-top',targetHandle:'target-bottom'};
  }, [nodePositions]);

  const edges = useMemo(() => {
    if (slide.id === 'overview') {
      return [
        {
          id:'overview-phase-1-2',
          source:'overview-phase-1',
          target:'overview-phase-2',
          sourceHandle:'source-right',
          targetHandle:'target-left',
          type:'smoothstep',
          pathOptions:{offset:32,borderRadius:16},
          markerEnd:{type:MarkerType.ArrowClosed,width:16,height:16},
          className:'overview-edge',
        },
        {
          id:'overview-phase-2-3',
          source:'overview-phase-2',
          target:'overview-phase-3',
          sourceHandle:'source-right',
          targetHandle:'target-left',
          type:'smoothstep',
          pathOptions:{offset:32,borderRadius:16},
          markerEnd:{type:MarkerType.ArrowClosed,width:16,height:16},
          className:'overview-edge',
        },
      ];
    }
    const phaseThreeMacroDependencies=new Set([
      'phase-2-complete-i33',
      'i36-i40',
      'i36-i41',
      'i36-i42',
      'i36-i47',
      'i36-i50',
      'i46-i50',
      'i49-i50',
    ]);
    const taskFlowEdges=taskEdges.filter(
      ([source,target]) => visibleIds.has(source) && visibleIds.has(target),
    ).filter(([source,target]) => slide.id !== 'phase-3' || !phaseThreeMacroDependencies.has(`${source}-${target}`),
    ).map(([source,target]) => ({
    id:`${source}-${target}`,
    source,
    target,
    ...edgeHandles(source,target),
    type:'smoothstep',
    pathOptions:{offset:28,borderRadius:12},
    markerEnd:{type:MarkerType.ArrowClosed,width:16,height:16},
    className:selectedId && !(connected.has(source) && connected.has(target)) ? 'faded' : '',
    style:connected.has(source) && connected.has(target) ? {strokeWidth:2.2} : undefined,
  }));
    if (slide.id === 'phase-3') {
      const macroEdge = (id,source,target,optional=false) => ({
        id,
        source,
        target,
        sourceHandle:'source-right',
        targetHandle:'target-left',
        type:'smoothstep',
        pathOptions:{offset:28,borderRadius:12},
        markerEnd:{type:MarkerType.ArrowClosed,width:16,height:16},
        className:optional ? 'phase-3-macro-edge optional' : 'phase-3-macro-edge',
      });
      return [
        ...taskFlowEdges,
        macroEdge('phase-3-start-sparse','phase-2-complete','phase-3-sparse-group'),
        macroEdge('phase-3-start-alternate','phase-2-complete','phase-3-alternate-group',true),
        macroEdge('phase-3-start-fft','phase-2-complete','phase-3-fft-group'),
        macroEdge('phase-3-start-tensor','phase-2-complete','phase-3-tensor-group'),
        macroEdge('phase-3-sparse-integration','phase-3-sparse-group','i50'),
        macroEdge('phase-3-sparse-lt-integration','phase-3-sparse-lt-group','i50',true),
        macroEdge('phase-3-alternate-integration','phase-3-alternate-group','i50',true),
        macroEdge('phase-3-fft-integration','phase-3-fft-group','i50'),
        macroEdge('phase-3-tensor-integration','phase-3-tensor-group','i50'),
      ];
    }
    if (slide.id !== 'phase-1') return taskFlowEdges;
    return [
      ...taskFlowEdges,
      {
        id:'phase-1-group-flow',
        source:'phase-1-core-group',
        target:'phase-1-integration-group',
        sourceHandle:'source-right',
        targetHandle:'target-left',
        type:'smoothstep',
        pathOptions:{offset:28,borderRadius:12},
        markerEnd:{type:MarkerType.ArrowClosed,width:16,height:16},
        className:'group-edge',
      },
    ];
  }, [selectedId,connected,visibleIds,edgeHandles,slide.id]);

  const onNodesChange = useCallback((changes) => {
    const moved = changes.filter((change) => change.type === 'position' && change.position);
    if (!moved.length) return;
    setNodePositions((current) => {
      const next={...current};
      moved.forEach((change) => { next[change.id]=[change.position.x,change.position.y]; });
      return next;
    });
  }, []);

  const selected = selectedId ? nodeById[selectedId] : null;
  const onNodeClick = useCallback((_,node) => {
    if (typeof node.data.phaseIndex === 'number') {
      const target=node.data.phaseIndex;
      setSlideDirection(target > activeSlide ? 'next' : 'previous');
      setSelectedId(null);
      setActiveSlide(target);
      return;
    }
    if (node.data.milestone) return;
    if (node.data.navigation) {
      const target = node.id === 'phase-1-complete'
        ? (slide.id === 'phase-1' ? 2 : 1)
        : (slide.id === 'phase-2' ? 3 : 2);
      setSlideDirection(target > activeSlide ? 'next' : 'previous');
      setSelectedId(null);
      setActiveSlide(target);
      return;
    }
    if (node.type === 'task') setSelectedId(node.id);
  }, [activeSlide,slide.id]);

  const goToSlide = useCallback((index) => {
    if (index < 0 || index >= slides.length || index === activeSlide) return;
    setSlideDirection(index > activeSlide ? 'next' : 'previous');
    setSelectedId(null);
    setActiveSlide(index);
  }, [activeSlide]);

  const openTaskReference = useCallback((id) => {
    const item=nodeById[id];
    if (!item) return;
    const targetSlide=slides.findIndex((candidate) => candidate.includes(item));
    if (targetSlide >= 0 && targetSlide !== activeSlide) {
      setSlideDirection(targetSlide > activeSlide ? 'next' : 'previous');
      setActiveSlide(targetSlide);
    }
    setSelectedId(id);
  }, [activeSlide,nodeById]);

  const openGlossary = (id) => {
    setGlossaryId(id);
    setTestCatalogOpen(false);
    setSpecCatalogOpen(false);
    setHelpOpen(true);
  };

  const openTestCatalog = (id='test-1') => {
    setCatalogTestId(id);
    setHelpOpen(false);
    setSpecCatalogOpen(false);
    setTestCatalogOpen(true);
  };

  const openSpecificationCatalog = (specification='architecture-component-model') => {
    const entry=typeof specification === 'string'
      ? specificationById[specification] ?? specificationByHref[specification]
      : specificationByHref[specification.href] ?? specificationById[specification.id];
    setCatalogSpecificationId(entry?.id ?? 'architecture-component-model');
    setHelpOpen(false);
    setTestCatalogOpen(false);
    setSpecCatalogOpen(true);
  };

  const linkedText = (value,glossarySeen=null) => value.split(richTextPattern).filter(Boolean).map((part,index) => {
    if (testById[part.toLowerCase()]) {
      const testId=part.toLowerCase();
      return <button className="test-reference" type="button" key={`${testId}-${index}`} onClick={() => openTestCatalog(testId)}>{part}</button>;
    }
    if (nodeById[part]) {
      return <button className="task-reference" type="button" key={`${part}-${index}`} onClick={() => openTaskReference(part)}>{part}</button>;
    }
    const aliasedTaskId=taskAliases[part.toLowerCase()];
    if (aliasedTaskId) {
      return <button className="task-reference" type="button" key={`${aliasedTaskId}-${index}`} onClick={() => openTaskReference(aliasedTaskId)}>{part}</button>;
    }
    const glossaryEntryId=glossaryAliases[part.toLowerCase()];
    if (glossaryEntryId) {
      if (glossarySeen?.has(glossaryEntryId)) return part;
      glossarySeen?.add(glossaryEntryId);
      return <button className="glossary-reference" type="button" key={`${glossaryEntryId}-${index}`} onClick={() => openGlossary(glossaryEntryId)}>{part}</button>;
    }
    return part;
  });

  const initializeViewport = useCallback((instance) => {
    requestAnimationFrame(async () => {
      await instance.fitView({padding:.15,maxZoom:1});
      if (slide.id !== 'overview') {
        await instance.zoomTo(Math.min(instance.getZoom()*2.5,2),{duration:0});
      }
    });
  }, [slide.id]);

  const nodeGlossaryTerms=new Set();

  return (
    <div className="app-shell">
      <header className="app-header">
        <div>
          <div className="eyebrow">ROCm math libraries</div>
          <h1>Interfaces delivery plan</h1>
          <p>Read-only dependency graph · select a node to inspect its contract</p>
        </div>
        <div className="header-actions">
          <button className="help-button" type="button" onClick={() => openSpecificationCatalog()}>Specifications</button>
          <button className="help-button" type="button" onClick={() => openTestCatalog()}>Tests</button>
          <button className="help-button" type="button" onClick={() => openGlossary('public-api')}>Help · definitions</button>
          <div className="header-legend">
            <span><i className="legend-dot ready" />Ready</span>
            <span><i className="legend-dot blocked" />Waiting</span>
            <span><i className="legend-dot later" />Later phase</span>
          </div>
        </div>
      </header>

      <div className="workspace">
        <div className="phase-review">
          <nav className="carousel-nav" aria-label="Plan phase navigation">
            <button className="carousel-arrow" onClick={() => goToSlide(activeSlide-1)} disabled={activeSlide===0} aria-label="Previous phase">←</button>
            <div className="carousel-copy">
              <div className="carousel-count">Section {activeSlide+1} of {slides.length}{activeSlide > 0 && <button className="overview-link" type="button" onClick={() => goToSlide(0)}>All phases</button>}</div>
              <h2>{slide.title}</h2>
              <p>{linkedText(slide.description)}</p>
            </div>
            <div className="carousel-dots">
              {slides.map((item,index) => <button key={item.id} className={index===activeSlide?'active':''} onClick={() => goToSlide(index)} aria-label={`Open ${item.title}`} />)}
            </div>
            <button className="carousel-arrow" onClick={() => goToSlide(activeSlide+1)} disabled={activeSlide===slides.length-1} aria-label="Next phase">→</button>
          </nav>
          <main className={`canvas-panel slide-${slideDirection}`} key={slide.id}>
            <ReactFlow
              nodes={nodes}
              edges={edges}
              nodeTypes={nodeTypes}
              onNodesChange={onNodesChange}
              onInit={initializeViewport}
              onNodeClick={onNodeClick}
              onPaneClick={() => setSelectedId(null)}
              nodesDraggable
              nodesConnectable={false}
              edgesFocusable={false}
              elementsSelectable
              colorMode="dark"
              minZoom={.3}
              maxZoom={2}
              proOptions={{hideAttribution:false}}
            >
              <Background gap={26} size={1} color="var(--grid-dot)" />
              <MiniMap pannable zoomable nodeColor={(node) => node.type === 'workGroup' ? '#1b242c' : `var(--${node.data.status})`} />
              <Controls showInteractive={false}>
                <ControlButton onClick={() => setNodePositions({...positions})} title="Reset node positions" aria-label="Reset node positions">↺</ControlButton>
              </Controls>
            </ReactFlow>
          </main>
        </div>

        <aside className={`inspector ${selected ? 'open' : ''}`}>
          {selected ? (
            <>
              <button className="inspector-close" onClick={() => setSelectedId(null)} aria-label="Close details">×</button>
              <div className="inspector-kicker">{selected.phase}</div>
              <div className="inspector-heading">
                <div><div className="inspector-id">{selected.id}</div><h2>{selected.label}</h2></div>
              </div>
              {statusLabel[selected.status] && <div className={`status-banner ${selected.status}`}><i />{statusLabel[selected.status]}</div>}
              <section><h3>Deliverable</h3><p>{linkedText(selected.summary,nodeGlossaryTerms)}</p></section>
              <section><h3>Implementation overview</h3><ul>{selected.details.map((item) => <li key={item}>{linkedText(item,nodeGlossaryTerms)}</li>)}</ul></section>
              {selected.specifications?.length > 0 && <section className="specification-section">
                <h3>Specification</h3>
                <ul>{selected.specifications.map((specification) => <li key={specification.name}><button className="specification-reference" type="button" onClick={() => openSpecificationCatalog(specification)}>{specification.name.replace(/\.md$/,'')} {specification.requirement}</button></li>)}</ul>
              </section>}
              {selected.tests?.length > 0 && <section>
                <h3>Task items</h3>
                <div className="test-plan">{selected.tests.map((test) => <details className={`test-row ${test.kind}`} key={test.itemId}>
                  <summary><span className="test-number">{test.itemId}</span> · {test.category}</summary>
                  <div className="test-content">
                    {(test.component || test.testIds?.length > 0) && <div className="item-labels">
                      {test.component && <button className="component-label" type="button" onClick={() => openGlossary(glossaryAliases[test.component.toLowerCase()] ?? test.component.toLowerCase())}>{test.component}</button>}
                      {test.testIds?.map((testId) => <button className="test-label" type="button" key={testId} onClick={() => openTestCatalog(testId)}>{testId}</button>)}
                    </div>}
                    <p><strong>Description:</strong> {linkedText(test.description,nodeGlossaryTerms)}</p>
                    <p className="test-outcome"><strong>Outcome:</strong> {linkedText(test.outcome,nodeGlossaryTerms)}</p>
                    {test.specification && <p className="test-specification"><strong>Specification:</strong> <button className="specification-reference" type="button" onClick={() => openSpecificationCatalog(test.specification)}>{test.specification.name.replace(/\.md$/,'')} {test.specification.requirement}</button></p>}
                  </div>
                </details>)}</div>
              </section>}
              <section>
                <h3>Definition of done</h3>
                <div className="done-plan">{selected.exit.map((item,index) => {
                  const testIds=item.testIds;
                  return <details className="done-row" key={item.doneId}>
                    <summary><span className="done-number">{index + 1}.</span> {item.title ?? item.text}</summary>
                    <div className="done-content">
                      {(testIds.length > 0 || item.specifications.length > 0) && <div className="item-labels">
                        {testIds.map((testId) => <button className="test-label" type="button" key={testId} onClick={() => openTestCatalog(testId)}>{testId}</button>)}
                        {item.specifications.map((specification) => <button className="spec-label" type="button" key={specification.name} onClick={() => openSpecificationCatalog(specification)}>{specification.name.replace(/\.md$/,'')} {specification.requirement}</button>)}
                      </div>}
                      <p>{linkedText(item.text,nodeGlossaryTerms)}</p>
                    </div>
                  </details>;
                })}</div>
              </section>
              {selected.pr && <section className="pr-section">
                <h3>PR</h3>
                <h4>{selected.pr.title}</h4>
                <p>{linkedText(selected.pr.description,nodeGlossaryTerms)}</p>
                <p className="path-assumption"><strong>Path assumption:</strong> <code>interfaces/</code> is located at <code>{'{projects,shared}/<component>/interfaces'}</code>.</p>
                {selected.pr.add?.length > 0 && <><h5>Files to add</h5><ul className="file-list">{selected.pr.add.map((file) => <li key={file}><code>{file}</code></li>)}</ul></>}
                {selected.pr.modify?.length > 0 && <><h5>Files to modify</h5><ul className="file-list">{selected.pr.modify.map((file) => <li key={file}><code>{file}</code></li>)}</ul></>}
              </section>}
            </>
          ) : (
            <div className="inspector-empty">
              <div className="empty-icon">⌁</div>
              <h2>Select a task</h2>
              <p>Choose a PR or spike to inspect its deliverable, prerequisites, and exit evidence.</p>
            </div>
          )}
        </aside>
      </div>
      {helpOpen && <div className="help-overlay" role="presentation" onMouseDown={() => setHelpOpen(false)}>
        <section className="help-dialog" role="dialog" aria-modal="true" aria-labelledby="help-title" onMouseDown={(event) => event.stopPropagation()}>
          <header className="help-header">
            <div>
              <div className="eyebrow">Plan vocabulary</div>
              <h2 id="help-title">Concepts and definitions</h2>
            </div>
            <button className="help-close" type="button" onClick={() => setHelpOpen(false)} aria-label="Close definitions">×</button>
          </header>
          <div className="help-layout">
            <nav className="help-terms" aria-label="Defined terms">
              {glossaryEntries.map((entry) => <button key={entry.id} type="button" className={entry.id===glossaryId?'active':''} onClick={() => setGlossaryId(entry.id)}>{entry.term}</button>)}
            </nav>
            <article className="help-definition">
              <div className="help-definition-label">Definition</div>
              <h3>{glossaryById[glossaryId].term}</h3>
              <p>{glossaryById[glossaryId].definition}</p>
              <dl>
                <dt>Includes</dt><dd>{glossaryById[glossaryId].includes}</dd>
                <dt>Example</dt><dd>{glossaryById[glossaryId].example}</dd>
                {glossaryById[glossaryId].relationship && <><dt>Relationship</dt><dd>{glossaryById[glossaryId].relationship}</dd></>}
              </dl>
              {glossaryById[glossaryId].breaks && <div className="help-breaks">
                <h4>Common breaks and how they surface</h4>
                <table>
                  <thead><tr><th>Change</th><th>How it surfaces</th></tr></thead>
                  <tbody>{glossaryById[glossaryId].breaks.map((item) => <tr key={item.change}><td>{item.change}</td><td>{item.surface}</td></tr>)}</tbody>
                </table>
              </div>}
              {glossaryById[glossaryId].steps && <div className="help-steps">
                <h4>Executed stages</h4>
                <ol>{glossaryById[glossaryId].steps.map((step) => <li key={step.term}><strong>{step.term}</strong><span>{step.definition}</span></li>)}</ol>
              </div>}
              {glossaryById[glossaryId].specification && <p className="help-specification"><strong>Specification:</strong> <button className="specification-reference" type="button" onClick={() => openSpecificationCatalog(glossaryById[glossaryId].specification)}>{glossaryById[glossaryId].specification.name.replace(/\.md$/,'')}</button></p>}
            </article>
          </div>
        </section>
      </div>}
      {testCatalogOpen && <div className="help-overlay" role="presentation" onMouseDown={() => setTestCatalogOpen(false)}>
        <section className="help-dialog test-catalog-dialog" role="dialog" aria-modal="true" aria-labelledby="test-catalog-title" onMouseDown={(event) => event.stopPropagation()}>
          <header className="help-header">
            <div>
              <div className="eyebrow">Verification catalog</div>
              <h2 id="test-catalog-title">Tests</h2>
            </div>
            <button className="help-close" type="button" onClick={() => setTestCatalogOpen(false)} aria-label="Close tests">×</button>
          </header>
          <div className="help-layout test-catalog-layout">
            <nav className="help-terms test-catalog-list" aria-label="Tests">
              {testCatalog.map((test) => <button key={test.testId} type="button" className={test.testId===catalogTestId?'active':''} onClick={() => setCatalogTestId(test.testId)}><span>{test.testId}</span>{test.category}</button>)}
            </nav>
            <article className="help-definition test-definition">
              <div className="help-definition-label">{testById[catalogTestId].testId} · defined by {testById[catalogTestId].definedByTaskId}</div>
              <h3>{testById[catalogTestId].category}</h3>
              <dl>
                <dt>Description</dt><dd>{testById[catalogTestId].description}</dd>
                <dt>Outcome</dt><dd>{testById[catalogTestId].outcome}</dd>
                <dt>Owner task</dt><dd><button className="task-reference" type="button" onClick={() => { setTestCatalogOpen(false); openTaskReference(testById[catalogTestId].taskId); }}>{testById[catalogTestId].taskId}</button> · {testById[catalogTestId].taskLabel}</dd>
                <dt>Initial state</dt><dd>{testById[catalogTestId].expectation ?? 'Not recorded'}</dd>
              </dl>
              {testById[catalogTestId].specification && <p className="help-specification"><strong>Specification:</strong> <button className="specification-reference" type="button" onClick={() => openSpecificationCatalog(testById[catalogTestId].specification)}>{testById[catalogTestId].specification.name.replace(/\.md$/,'')} {testById[catalogTestId].specification.requirement}</button></p>}
            </article>
          </div>
        </section>
      </div>}
      {specCatalogOpen && <div className="help-overlay" role="presentation" onMouseDown={() => setSpecCatalogOpen(false)}>
        <section className="help-dialog specification-catalog-dialog" role="dialog" aria-modal="true" aria-labelledby="specification-catalog-title" onMouseDown={(event) => event.stopPropagation()}>
          <header className="help-header">
            <div>
              <div className="eyebrow">Planning catalog</div>
              <h2 id="specification-catalog-title">Specifications</h2>
            </div>
            <button className="help-close" type="button" onClick={() => setSpecCatalogOpen(false)} aria-label="Close specifications">×</button>
          </header>
          <div className="help-layout specification-catalog-layout">
            <nav className="help-terms specification-catalog-list" aria-label="Specifications">
              {specificationEntries.map((specification) => <button key={specification.id} type="button" className={specification.id===catalogSpecificationId?'active':''} onClick={() => setCatalogSpecificationId(specification.id)}><span>{specification.kind}</span>{specification.name}</button>)}
            </nav>
            <article className="help-definition specification-definition">
              <div className="help-definition-label">{specificationById[catalogSpecificationId].kind}</div>
              <h3>{specificationById[catalogSpecificationId].title}</h3>
              <p>{specificationById[catalogSpecificationId].description}</p>
              <dl>
                <dt>Catalog name</dt><dd>{specificationById[catalogSpecificationId].name}</dd>
                <dt>Review page</dt><dd><code>{`specifications/${specificationById[catalogSpecificationId].id}.html`}</code></dd>
              </dl>
              <p className="help-specification"><a className="spec-review-link" href={`specifications/${specificationById[catalogSpecificationId].id}.html`} target="_blank" rel="noreferrer">Open formatted review page</a></p>
            </article>
          </div>
        </section>
      </div>}
    </div>
  );
}

createRoot(document.getElementById('root')).render(
  <StrictMode><ReactFlowProvider><App /></ReactFlowProvider></StrictMode>,
);
