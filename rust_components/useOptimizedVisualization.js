/*
 * Optimized Visualization Hook (useOptimizedVisualization.js)
 * 
 * PURPOSE:
 * This custom React hook provides an optimized approach to rendering complex
 * visualizations with high-frequency data updates. It intelligently manages
 * when to re-render and chooses the most efficient rendering technique.
 *
 * KEY FUNCTIONS:
 * - Provides a unified API for both SVG and Canvas-based visualizations
 * - Implements performance optimizations like throttling and update thresholds
 * - Manages visualization lifecycle (initialization, updates, cleanup)
 * - Handles rendering and scaling for both SVG and Canvas contexts
 * - Efficiently updates only what's changed instead of re-rendering everything
 */

// useOptimizedVisualization.js
import { useRef, useEffect, useState } from 'react';
import * as d3 from 'd3';

export function useOptimizedVisualization(data, options = {}) {
  const {
    width = 800,
    height = 600,
    updateThreshold = 0.05,
    throttleTime = 33,  // ~30fps
    renderType = 'canvas'  // 'svg' or 'canvas'
  } = options;
  
  const containerRef = useRef(null);
  const rendererRef = useRef(null);
  const dataRef = useRef(null);
  const lastRenderTimeRef = useRef(0);
  const [isReady, setIsReady] = useState(false);
  
  // Initialize visualization
  useEffect(() => {
    if (!containerRef.current) return;
    
    // Clear previous renderer
    if (rendererRef.current) {
      containerRef.current.innerHTML = '';
    }
    
    // Create new renderer based on type
    if (renderType === 'canvas') {
      const canvas = document.createElement('canvas');
      canvas.width = width;
      canvas.height = height;
      containerRef.current.appendChild(canvas);
      
      rendererRef.current = {
        type: 'canvas',
        context: canvas.getContext('2d'),
        element: canvas
      };
    } else {
      const svg = d3.select(containerRef.current)
        .append('svg')
        .attr('width', width)
        .attr('height', height);
      
      rendererRef.current = {
        type: 'svg',
        element: svg,
        groups: {
          nodes: svg.append('g').attr('class', 'nodes'),
          links: svg.append('g').attr('class', 'links'),
          labels: svg.append('g').attr('class', 'labels')
        }
      };
    }
    
    dataRef.current = data;
    setIsReady(true);
    
    // Initial render
    renderVisualization(data);
    
    return () => {
      // Cleanup
      if (containerRef.current) {
        containerRef.current.innerHTML = '';
      }
    };
  }, [width, height, renderType]);
  
  // Update visualization when data changes
  useEffect(() => {
    if (!isReady || !data) return;
    
    // Check if we should update based on threshold
    const shouldUpdate = checkUpdateThreshold(dataRef.current, data, updateThreshold);
    
    if (shouldUpdate) {
      const now = performance.now();
      
      // Throttle updates
      if (now - lastRenderTimeRef.current > throttleTime) {
        renderVisualization(data);
        lastRenderTimeRef.current = now;
        dataRef.current = data;
      } else {
        // Schedule update for later if we're throttling
        const delayTime = throttleTime - (now - lastRenderTimeRef.current);
        setTimeout(() => {
          renderVisualization(data);
          lastRenderTimeRef.current = performance.now();
          dataRef.current = data;
        }, delayTime);
      }
    }
  }, [data, updateThreshold, throttleTime, isReady]);
  
  // Helper functions
  const checkUpdateThreshold = (oldData, newData, threshold) => {
    if (!oldData) return true;
    
    // Implement your logic to determine if the change is significant
    // This will depend on your data structure
    // Example for a simple metric:
    const oldValue = oldData.metric;
    const newValue = newData.metric;
    
    return Math.abs(newValue - oldValue) > threshold;
  };
  
  const renderVisualization = (data) => {
    if (!rendererRef.current || !data) return;
    
    // Choose rendering approach based on renderer type
    if (rendererRef.current.type === 'canvas') {
      renderCanvasVisualization(data, rendererRef.current.context, width, height);
    } else {
      renderSvgVisualization(data, rendererRef.current, width, height);
    }
  };
  
  // Canvas rendering is more efficient for large numbers of objects
  const renderCanvasVisualization = (data, ctx, width, height) => {
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    // Implement your specific visualization logic here
    // Example for a simple node network:
    ctx.fillStyle = '#333';
    ctx.strokeStyle = '#999';
    ctx.lineWidth = 1;
    
    // Draw links
    if (data.links) {
      data.links.forEach(link => {
        ctx.beginPath();
        ctx.moveTo(link.source.x, link.source.y);
        ctx.lineTo(link.target.x, link.target.y);
        ctx.stroke();
      });
    }
    
    // Draw nodes
    if (data.nodes) {
      data.nodes.forEach(node => {
        ctx.beginPath();
        ctx.arc(node.x, node.y, node.radius || 5, 0, Math.PI * 2);
        ctx.fill();
      });
    }
  };
  
  // SVG rendering is better for interactive elements and smaller visualizations
  const renderSvgVisualization = (data, renderer, width, height) => {
    const { element, groups } = renderer;
    
    // Implement your specific visualization logic here
    // Example for a simple node network:
    
    // Update links
    if (data.links && groups.links) {
      const links = groups.links
        .selectAll('line')
        .data(data.links, d => `${d.source.id}-${d.target.id}`);
      
      links.exit().remove();
      
      links.enter()
        .append('line')
        .merge(links)
        .attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x)
        .attr('y2', d => d.target.y)
        .attr('stroke', '#999')
        .attr('stroke-width', 1);
    }
    
    // Update nodes
    if (data.nodes && groups.nodes) {
      const nodes = groups.nodes
        .selectAll('circle')
        .data(data.nodes, d => d.id);
      
      nodes.exit().remove();
      
      nodes.enter()
        .append('circle')
        .merge(nodes)
        .attr('cx', d => d.x)
        .attr('cy', d => d.y)
        .attr('r', d => d.radius || 5)
        .attr('fill', '#333');
    }
  };
  
  return { containerRef, isReady };
}
