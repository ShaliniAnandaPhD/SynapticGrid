```jsx
/**
 * Waste Heatmap Component (WasteMap.jsx)
 * --------------------------------------
 * A React-based map visualization that displays waste generation hotspots
 * based on bin sensor data. Supports smart waste management systems with
 * interactive filters, real-time updates, and analytics.
 */

import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { 
  MapContainer, TileLayer, Marker, Popup, Circle, LayersControl, 
  ZoomControl, ScaleControl, useMap
} from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';
import HeatmapLayer from 'react-leaflet-heatmap-layer';
import { 
  Box, Typography, Paper, Card, CardContent, Grid, FormControl, 
  InputLabel, Select, MenuItem, Divider, Slider, Switch,
  FormControlLabel, Button, Chip, Stack, IconButton, TextField,
  Drawer, List, ListItem, ListItemText, ListItemIcon, Collapse,
  Alert, Tooltip, CircularProgress, Tabs, Tab, useTheme, AppBar,
  Toolbar, Badge
} from '@mui/material';
import {
  FilterAlt as FilterIcon,
  Refresh as RefreshIcon,
  Delete as DeleteIcon,
  Place as PlaceIcon,
  CalendarToday as CalendarIcon,
  Info as InfoIcon,
  Map as MapIcon,
  Timeline as TimelineIcon,
  BarChart as BarChartIcon,
  PieChart as PieChartIcon,
  NavigateNext as NavigateNextIcon,
  NavigateBefore as NavigateBeforeIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  Settings as SettingsIcon,
  Layers as LayersIcon,
  LocalShipping as TruckIcon,
  RotateLeft as HistoryIcon,
  Search as SearchIcon,
  Notifications as NotificationsIcon,
  Download as DownloadIcon
} from '@mui/icons-material';
import { format, subDays, addDays, parseISO, isAfter, isBefore } from 'date-fns';
import axios from 'axios';
import { 
  LineChart, Line, BarChart, Bar, PieChart, Pie, Cell, 
  XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, 
  Legend, ResponsiveContainer, AreaChart, Area
} from 'recharts';
import DatePicker from 'react-datepicker';
import 'react-datepicker/dist/react-datepicker.css';
import { debounce } from 'lodash';
import { saveAs } from 'file-saver';

// Fix for Leaflet icon issues with webpack
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-shadow.png',
});

// Waste type definitions with colors
const WASTE_TYPES = [
  { id: 'general', name: 'General Waste', color: '#607D8B' },
  { id: 'recyclable', name: 'Recyclables', color: '#4CAF50' },
  { id: 'organic', name: 'Organic Waste', color: '#8BC34A' },
  { id: 'paper', name: 'Paper', color: '#00BCD4' },
  { id: 'glass', name: 'Glass', color: '#9C27B0' },
  { id: 'hazardous', name: 'Hazardous', color: '#F44336' },
  { id: 'electronic', name: 'Electronic', color: '#FF9800' },
  { id: 'textile', name: 'Textile', color: '#9E9E9E' }
];

// Collection zone definitions
const COLLECTION_ZONES = [
  { id: 'north', name: 'North Zone' },
  { id: 'south', name: 'South Zone' },
  { id: 'east', name: 'East Zone' },
  { id: 'west', name: 'West Zone' },
  { id: 'central', name: 'Central Zone' }
];

// Custom bin icons by fill level
const createBinIcon = (fillLevel, wasteType = 'general') => {
  // Determine color based on fill level
  const fillColor = fillLevel < 30 ? '#4CAF50' :  // Green for low fill
                   fillLevel < 70 ? '#FFC107' :  // Yellow for medium fill
                   '#F44336';                    // Red for high fill
  
  // Get the waste type color for the border
  const wasteTypeObj = WASTE_TYPES.find(type => type.id === wasteType) || WASTE_TYPES[0];
  const borderColor = wasteTypeObj.color;
                   
  return new L.DivIcon({
    className: 'custom-bin-icon',
    html: `
      <div style="
        background-color: ${fillColor}; 
        width: 20px; 
        height: 20px; 
        border-radius: 50%; 
        border: 3px solid ${borderColor};
        display: flex; 
        justify-content: center; 
        align-items: center;
        box-shadow: 0 2px 5px rgba(0,0,0,0.3);
      "></div>
    `,
    iconSize: [26, 26],
    iconAnchor: [13, 13],
    popupAnchor: [0, -15]
  });
};

// Truck icon for collection vehicles
const truckIcon = new L.Icon({
  iconUrl: 'https://cdn-icons-png.flaticon.com/512/3026/3026635.png',
  iconSize: [32, 32],
  iconAnchor: [16, 16],
  popupAnchor: [0, -16],
});

/**
 * Tab Panel component for organizing the dashboard
 */
const TabPanel = ({ children, value, index, ...other }) => {
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`waste-map-tabpanel-${index}`}
      aria-labelledby={`waste-map-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 2 }}>{children}</Box>
      )}
    </div>
  );
};

/**
 * Legend component for the map
 */
const MapLegend = ({ visible = true }) => {
  if (!visible) return null;
  
  return (
    <Card 
      sx={{ 
        position: 'absolute', 
        bottom: 20, 
        right: 20, 
        zIndex: 1000, 
        width: 220,
        opacity: 0.9
      }}
    >
      <CardContent sx={{ p: 1, '&:last-child': { pb: 1 } }}>
        <Typography variant="subtitle2" gutterBottom>Fill Level</Typography>
        
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 0.5 }}>
          <Box 
            sx={{ 
              width: 16, 
              height: 16, 
              borderRadius: '50%', 
              backgroundColor: '#4CAF50', 
              mr: 1,
              border: '2px solid white',
              boxShadow: '0 1px 3px rgba(0,0,0,0.2)'
            }} 
          />
          <Typography variant="body2">Low (0-30%)</Typography>
        </Box>
        
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 0.5 }}>
          <Box 
            sx={{ 
              width: 16, 
              height: 16, 
              borderRadius: '50%', 
              backgroundColor: '#FFC107', 
              mr: 1,
              border: '2px solid white',
              boxShadow: '0 1px 3px rgba(0,0,0,0.2)'
            }} 
          />
          <Typography variant="body2">Medium (30-70%)</Typography>
        </Box>
        
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
          <Box 
            sx={{ 
              width: 16, 
              height: 16, 
              borderRadius: '50%', 
              backgroundColor: '#F44336', 
              mr: 1,
              border: '2px solid white',
              boxShadow: '0 1px 3px rgba(0,0,0,0.2)'
            }} 
          />
          <Typography variant="body2">High (70-100%)</Typography>
        </Box>
        
        <Divider sx={{ my: 1 }} />
        
        <Typography variant="subtitle2" gutterBottom>Waste Types</Typography>
        
        {WASTE_TYPES.slice(0, 4).map(type => (
          <Box key={type.id} sx={{ display: 'flex', alignItems: 'center', mb: 0.5 }}>
            <Box 
              sx={{ 
                width: 12, 
                height: 12, 
                borderRadius: '50%', 
                backgroundColor: type.color, 
                mr: 1 
              }} 
            />
            <Typography variant="body2">{type.name}</Typography>
          </Box>
        ))}
        
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <Box 
            sx={{ 
              width: 20, 
              height: 14, 
              mr: 1,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }} 
          >
            <img 
              src="https://cdn-icons-png.flaticon.com/512/3026/3026635.png" 
              alt="Collection Vehicle" 
              style={{ width: '100%', height: 'auto' }}
            />
          </Box>
          <Typography variant="body2">Collection Vehicle</Typography>
        </Box>
      </CardContent>
    </Card>
  );
};

/**
 * Filter controls component
 */
const FilterControls = ({ 
  filters, 
  setFilters, 
  dateRange, 
  setDateRange, 
  onRefresh,
  onReset
}) => {
  const [expanded, setExpanded] = useState(true);
  
  // Handle filter changes
  const handleFilterChange = (field, value) => {
    setFilters(prev => ({
      ...prev,
      [field]: value
    }));
  };
  
  return (
    <Paper 
      elevation={3} 
      sx={{ 
        p: 2, 
        mb: 2, 
        borderRadius: 2,
        transition: 'all 0.3s ease'
      }}
    >
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
        <Typography variant="h6" component="h2" sx={{ display: 'flex', alignItems: 'center' }}>
          <FilterIcon sx={{ mr: 1 }} />
          Filters & Controls
        </Typography>
        
        <Box>
          <IconButton onClick={onRefresh} color="primary" size="small" sx={{ mr: 1 }}>
            <RefreshIcon />
          </IconButton>
          <IconButton onClick={() => setExpanded(!expanded)} size="small">
            {expanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
          </IconButton>
        </Box>
      </Box>
      
      <Collapse in={expanded}>
        <Grid container spacing={2} sx={{ mt: 1 }}>
          {/* Date Range Filter */}
          <Grid item xs={12} md={6}>
            <Card variant="outlined" sx={{ p: 1 }}>
              <Typography variant="subtitle2" sx={{ mb: 1, display: 'flex', alignItems: 'center' }}>
                <CalendarIcon fontSize="small" sx={{ mr: 1 }} />
                Date Range
              </Typography>
              <DatePicker
                selectsRange={true}
                startDate={dateRange.startDate}
                endDate={dateRange.endDate}
                onChange={(dates) => {
                  const [start, end] = dates;
                  setDateRange({ startDate: start, endDate: end || start });
                }}
                customInput={
                  <TextField 
                    fullWidth 
                    size="small" 
                    InputProps={{
                      endAdornment: <CalendarIcon color="action" />
                    }}
                  />
                }
                maxDate={new Date()}
              />
            </Card>
          </Grid>
          
          {/* Waste Type Filter */}
          <Grid item xs={12} sm={6} md={3}>
            <Card variant="outlined" sx={{ p: 1 }}>
              <Typography variant="subtitle2" sx={{ mb: 1 }}>Waste Type</Typography>
              <FormControl fullWidth size="small">
                <Select
                  value={filters.wasteType}
                  onChange={(e) => handleFilterChange('wasteType', e.target.value)}
                  displayEmpty
                >
                  <MenuItem value="">All Types</MenuItem>
                  {WASTE_TYPES.map((type) => (
                    <MenuItem key={type.id} value={type.id}>
                      <Box sx={{ display: 'flex', alignItems: 'center' }}>
                        <Box 
                          sx={{ 
                            width: 12, 
                            height: 12, 
                            borderRadius: '50%', 
                            backgroundColor: type.color,
                            mr: 1 
                          }} 
                        />
                        {type.name}
                      </Box>
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Card>
          </Grid>
          
          {/* Collection Zone Filter */}
          <Grid item xs={12} sm={6} md={3}>
            <Card variant="outlined" sx={{ p: 1 }}>
              <Typography variant="subtitle2" sx={{ mb: 1 }}>Collection Zone</Typography>
              <FormControl fullWidth size="small">
                <Select
                  value={filters.zone}
                  onChange={(e) => handleFilterChange('zone', e.target.value)}
                  displayEmpty
                >
                  <MenuItem value="">All Zones</MenuItem>
                  {COLLECTION_ZONES.map((zone) => (
                    <MenuItem key={zone.id} value={zone.id}>
                      {zone.name}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Card>
          </Grid>
          
          {/* Fill Level Range Slider */}
          <Grid item xs={12}>
            <Card variant="outlined" sx={{ p: 1, pb: 2 }}>
              <Typography variant="subtitle2" sx={{ mb: 1 }}>
                Fill Level Range (%)
              </Typography>
              <Box sx={{ px: 2 }}>
                <Slider
                  value={filters.fillLevelRange}
                  onChange={(e, newValue) => handleFilterChange('fillLevelRange', newValue)}
                  valueLabelDisplay="auto"
                  min={0}
                  max={100}
                  marks={[
                    { value: 0, label: '0%' },
                    { value: 25, label: '25%' },
                    { value: 50, label: '50%' },
                    { value: 75, label: '75%' },
                    { value: 100, label: '100%' },
                  ]}
                />
              </Box>
            </Card>
          </Grid>
          
          {/* Control Buttons */}
          <Grid item xs={12}>
            <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 1 }}>
              <Button
                variant="outlined"
                startIcon={<DeleteIcon />}
                onClick={onReset}
                size="small"
                sx={{ mr: 1 }}
              >
                Reset Filters
              </Button>
              <Button
                variant="contained"
                startIcon={<RefreshIcon />}
                onClick={onRefresh}
                size="small"
              >
                Refresh Data
              </Button>
            </Box>
          </Grid>
        </Grid>
      </Collapse>
    </Paper>
  );
};

/**
 * Statistics summary component
 */
const StatisticsSummary = ({ data, isLoading }) => {
  // Calculate statistics
  const stats = useMemo(() => {
    if (!data || data.length === 0) {
      return {
        totalBins: 0,
        binsNeedingCollection: 0,
        averageFillLevel: 0,
        mostCommonWasteType: 'N/A'
      };
    }
    
    // Total bins count
    const totalBins = data.length;
    
    // Bins needing collection (fill level > 70%)
    const binsNeedingCollection = data.filter(bin => bin.fillLevel > 70).length;
    
    // Average fill level
    const averageFillLevel = data.reduce((sum, bin) => sum + bin.fillLevel, 0) / totalBins;
    
    // Most common waste type
    const wasteTypeCounts = data.reduce((counts, bin) => {
      counts[bin.wasteType] = (counts[bin.wasteType] || 0) + 1;
      return counts;
    }, {});
    
    const mostCommonWasteType = Object.entries(wasteTypeCounts)
      .sort((a, b) => b[1] - a[1])[0]?.[0] || 'N/A';
      
    const mostCommonName = WASTE_TYPES.find(type => type.id === mostCommonWasteType)?.name || mostCommonWasteType;
    
    return {
      totalBins,
      binsNeedingCollection,
      averageFillLevel,
      mostCommonWasteType: mostCommonName
    };
  }, [data]);
  
  if (isLoading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
        <CircularProgress />
      </Box>
    );
  }
  
  return (
    <Card variant="outlined">
      <CardContent>
        <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
          <InfoIcon sx={{ mr: 1 }} />
          Quick Statistics
        </Typography>
        
        <Grid container spacing={2} sx={{ mt: 1 }}>
          <Grid item xs={6} md={3}>
            <Box sx={{ textAlign: 'center', p: 1 }}>
              <Typography variant="h4" color="primary">{stats.totalBins}</Typography>
              <Typography variant="body2" color="textSecondary">Total Bins</Typography>
            </Box>
          </Grid>
          
          <Grid item xs={6} md={3}>
            <Box sx={{ textAlign: 'center', p: 1 }}>
              <Typography 
                variant="h4" 
                color={stats.binsNeedingCollection > 0 ? "error" : "success"}
              >
                {stats.binsNeedingCollection}
              </Typography>
              <Typography variant="body2" color="textSecondary">Bins Needing Collection</Typography>
            </Box>
          </Grid>
          
          <Grid item xs={6} md={3}>
            <Box sx={{ textAlign: 'center', p: 1 }}>
              <Typography 
                variant="h4" 
                color={
                  stats.averageFillLevel < 30 ? "success" : 
                  stats.averageFillLevel < 70 ? "warning" : "error"
                }
              >
                {stats.averageFillLevel.toFixed(1)}%
              </Typography>
              <Typography variant="body2" color="textSecondary">Average Fill Level</Typography>
            </Box>
          </Grid>
          
          <Grid item xs={6} md={3}>
            <Box sx={{ textAlign: 'center', p: 1 }}>
              <Typography variant="h6" color="info.main">{stats.mostCommonWasteType}</Typography>
              <Typography variant="body2" color="textSecondary">Most Common Waste Type</Typography>
            </Box>
          </Grid>
        </Grid>
      </CardContent>
    </Card>
  );
};

/**
 * Analytics Charts Component
 */
const AnalyticsCharts = ({ data, isLoading }) => {
  const [activeChart, setActiveChart] = useState(0);
  
  // Process data for charts
  const chartData = useMemo(() => {
    if (!data || data.length === 0) {
      return {
        fillLevelDistribution: [],
        wasteTypeDistribution: [],
        fillLevelByZone: []
      };
    }
    
    // Fill level distribution
    const fillLevelCounts = {
      'Low (0-30%)': 0,
      'Medium (30-70%)': 0,
      'High (70-100%)': 0
    };
    
    data.forEach(bin => {
      if (bin.fillLevel < 30) fillLevelCounts['Low (0-30%)']++;
      else if (bin.fillLevel < 70) fillLevelCounts['Medium (30-70%)']++;
      else fillLevelCounts['High (70-100%)']++;
    });
    
    const fillLevelDistribution = Object.entries(fillLevelCounts).map(([name, value]) => ({ name, value }));
    
    // Waste type distribution
    const wasteTypeCounts = {};
    data.forEach(bin => {
      wasteTypeCounts[bin.wasteType] = (wasteTypeCounts[bin.wasteType] || 0) + 1;
    });
    
    const wasteTypeDistribution = Object.entries(wasteTypeCounts).map(([id, value]) => {
      const wasteType = WASTE_TYPES.find(type => type.id === id) || { id, name: id, color: '#999' };
      return { 
        name: wasteType.name, 
        value, 
        color: wasteType.color 
      };
    });
    
    // Fill level by zone
    const zoneData = {};
    data.forEach(bin => {
      if (!zoneData[bin.zone]) {
        zoneData[bin.zone] = { totalFill: 0, count: 0 };
      }
      zoneData[bin.zone].totalFill += bin.fillLevel;
      zoneData[bin.zone].count++;
    });
    
    const fillLevelByZone = Object.entries(zoneData).map(([id, zoneStat]) => {
      const zone = COLLECTION_ZONES.find(z => z.id === id) || { id, name: id };
      return {
        name: zone.name,
        fillLevel: zoneStat.totalFill / zoneStat.count
      };
    });
    
    return {
      fillLevelDistribution,
      wasteTypeDistribution,
      fillLevelByZone
    };
  }, [data]);
  
  if (isLoading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
        <CircularProgress />
      </Box>
    );
  }
  
  return (
    <Card variant="outlined">
      <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
        <Tabs 
          value={activeChart} 
          onChange={(e, newValue) => setActiveChart(newValue)}
          variant="fullWidth"
        >
          <Tab 
            icon={<PieChartIcon fontSize="small" />} 
            label="Fill Levels" 
            id="waste-chart-tab-0" 
          />
          <Tab 
            icon={<PieChartIcon fontSize="small" />} 
            label="Waste Types" 
            id="waste-chart-tab-1" 
          />
          <Tab 
            icon={<BarChartIcon fontSize="small" />} 
            label="By Zone" 
            id="waste-chart-tab-2" 
          />
        </Tabs>
      </Box>
      
      {/* Fill Level Distribution Chart */}
      <TabPanel value={activeChart} index={0}>
        <Typography variant="subtitle1" align="center" gutterBottom>
          Fill Level Distribution
        </Typography>
        
        <Box sx={{ height: 300, width: '100%' }}>
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={chartData.fillLevelDistribution}
                cx="50%"
                cy="50%"
                innerRadius={60}
                outerRadius={100}
                paddingAngle={2}
                dataKey="value"
                label={({ name, percent }) => `${name} (${(percent * 100).toFixed(0)}%)`}
              >
                <Cell key="low" fill="#4CAF50" />
                <Cell key="medium" fill="#FFC107" />
                <Cell key="high" fill="#F44336" />
              </Pie>
              <RechartsTooltip formatter={(value) => [`${value} bins`, 'Count']} />
            </PieChart>
          </ResponsiveContainer>
        </Box>
      </TabPanel>
      
      {/* Waste Type Distribution Chart */}
      <TabPanel value={activeChart} index={1}>
        <Typography variant="subtitle1" align="center" gutterBottom>
          Waste Type Distribution
        </Typography>
        
        <Box sx={{ height: 300, width: '100%' }}>
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={chartData.wasteTypeDistribution}
                cx="50%"
                cy="50%"
                outerRadius={100}
                paddingAngle={2}
                dataKey="value"
                label={({ name, percent }) => `${name} (${(percent * 100).toFixed(0)}%)`}
              >
                {chartData.wasteTypeDistribution.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Legend />
              <RechartsTooltip formatter={(value) => [`${value} bins`, 'Count']} />
            </PieChart>
          </ResponsiveContainer>
        </Box>
      </TabPanel>
      
      {/* Fill Level by Zone Chart */}
      <TabPanel value={activeChart} index={2}>
        <Typography variant="subtitle1" align="center" gutterBottom>
          Average Fill Level by Zone
        </Typography>
        
        <Box sx={{ height: 300, width: '100%' }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData.fillLevelByZone}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis 
                label={{ value: 'Fill Level (%)', angle: -90, position: 'insideLeft' }} 
                domain={[0, 100]} 
              />
              <RechartsTooltip formatter={(value) => [`${value.toFixed(1)}%`, 'Fill Level']} />
              <Bar 
                dataKey="fillLevel" 
                name="Average Fill Level" 
                barSize={40}
                fill="#8884d8"
              >
                {chartData.fillLevelByZone.map((entry, index) => (
                  <Cell 
                    key={`cell-${index}`} 
                    fill={
                      entry.fillLevel < 30 ? '#4CAF50' : 
                      entry.fillLevel < 70 ? '#FFC107' : '#F44336'
                    } 
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Box>
      </TabPanel>
    </Card>
  );
};

/**
 * Bin Details Panel component
 */
const BinDetailsPanel = ({ selectedBin, onClose }) => {
  if (!selectedBin) return null;
  
  // Find the waste type info
  const wasteType = WASTE_TYPES.find(type => type.id === selectedBin.wasteType) || WASTE_TYPES[0];
  
  // Fill level styling
  const fillLevelColor = 
    selectedBin.fillLevel < 30 ? '#4CAF50' : 
    selectedBin.fillLevel < 70 ? '#FFC107' : '#F44336';
  
  // Last collected date
  const lastCollected = selectedBin.lastCollected ? 
    format(new Date(selectedBin.lastCollected), 'MMM dd, yyyy HH:mm') : 
    'Never';
  
  return (
    <Drawer
      anchor="right"
      open={Boolean(selectedBin)}
      onClose={onClose}
      PaperProps={{
        sx: { width: { xs: '100%', sm: 400 }, p: 2 }
      }}
    >
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6">Bin Details</Typography>
        <IconButton onClick={onClose} size="small">
          <DeleteIcon fontSize="small" />
        </IconButton>
      </Box>
      
      <Card variant="outlined" sx={{ mb: 2 }}>
        <CardContent>
          <Typography variant="h5" gutterBottom>
            {selectedBin.name || `Bin #${selectedBin.id}`}
          </Typography>
          
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
            <PlaceIcon fontSize="small" sx={{ mr: 1, color: 'text.secondary' }} />
            <Typography variant="body2" color="text.secondary">
              {selectedBin.location || `${selectedBin.lat.toFixed(6)}, ${selectedBin.lng.toFixed(6)}`}
            </Typography>
          </Box>
          
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
            <Box 
              sx={{ 
                width: 12, 
                height: 12, 
                borderRadius: '50%', 
                backgroundColor: wasteType.color,
                mr: 1 
              }} 
            />
            <Typography variant="body2" color="text.secondary">
              {wasteType.name}
            </Typography>
          </Box>
          
          <Divider sx={{ my: 2 }} />
          
          <Typography variant="subtitle2" gutterBottom>Fill Level</Typography>
          <Box sx={{ position: 'relative', height: 20, backgroundColor: '#f5f5f5', borderRadius: 1, mb: 1 }}>
            <Box 
              sx={{ 
                position: 'absolute', 
                left: 0, 
                top: 0, 
                height: '100%', 
                width: `${selectedBin.fillLevel}%`,
                backgroundColor: fillLevelColor,
                borderRadius: 1,
                transition: 'width 0.5s ease'
              }} 
            />
          </Box>
          <Typography variant="h4" align="center" sx={{ color: fillLevelColor, my: 1 }}>
            {selectedBin.fillLevel}%
          </Typography>
          
          <Divider sx={{ my: 2 }} />
          
          <Grid container spacing={2}>
            <Grid item xs={6}>
              <Typography variant="subtitle2">Last Collected</Typography>
              <Typography variant="body2">{lastCollected}</Typography>
            </Grid>
            <Grid item xs={6}>
              <Typography variant="subtitle2">Zone</Typography>
              <Typography variant="body2">
                {COLLECTION_ZONES.find(zone => zone.id === selectedBin.zone)?.name || selectedBin.zone}
              </Typography>
            </Grid>
            <Grid item xs={6}>
              <Typography variant="subtitle2">Capacity</Typography>
              <Typography variant="body2">{selectedBin.capacity || '120 L'}</Typography>
            </Grid>
            <Grid item xs={6}>
              <Typography variant="subtitle2">Status</Typography>
              <Chip 
                label={selectedBin.fillLevel > 70 ? "Needs Collection" : "OK"} 
                color={selectedBin.fillLevel > 70 ? "error" : "success"}
                size="small"
              />
            </Grid>
          </Grid>
        </CardContent>
      </Card>
      
      <Card variant="outlined">
        <CardContent>
          <Typography variant="subtitle1" gutterBottom>Fill Level History</Typography>
          <Box sx={{ height: 200, width: '100%' }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={selectedBin.history || [
                  // Fallback dummy data if no history available
                  { date: '6d ago', level: 10 },
                  { date: '5d ago', level: 25 },
                  { date: '4d ago', level: 40 },
                  { date: '3d ago', level: 55 },
                  { date: '2d ago', level: 70 },
                  { date: '1d ago', level: 85 },
                  { date: 'now', level: selectedBin.fillLevel }
                ]}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis domain={[0, 100]} label={{ value: 'Fill %', angle: -90, position: 'insideLeft' }} />
                <RechartsTooltip formatter={(value) => [`${value}%`, 'Fill Level']} />
                <Line 
                  type="monotone" 
                  dataKey="level" 
                  stroke="#8884d8" 
                  activeDot={{ r: 8 }} 
                  strokeWidth={2}
                />
              </LineChart>
            </ResponsiveContainer>
          </Box>
          
          <Box sx={{ mt: 2, display: 'flex', justifyContent: 'center' }}>
            <Button 
              variant="outlined" 
              startIcon={<TruckIcon />}
              color="primary"
              size="small"
            >
              Schedule Collection
            </Button>
          </Box>
        </CardContent>
      </Card>
    </Drawer>
  );
};

/**
 * Main WasteMap Component
 */
const WasteMap = ({ apiUrl = '/api/waste-bins', initialCenter = [51.505, -0.09], initialZoom = 13 }) => {
  // State for map data and UI controls
  const [bins, setBins] = useState([]);
  const [vehicles, setVehicles] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedBin, setSelectedBin] = useState(null);
  const [activeTab, setActiveTab] = useState(0);
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [mapLegendVisible, setMapLegendVisible] = useState(true);
  const [lastUpdated, setLastUpdated] = useState(null);
  
  // Filters state
  const [filters, setFilters] = useState({
    wasteType: '',
    zone: '',
    fillLevelRange: [0, 100]
  });
  
  // Date range filter
  const [dateRange, setDateRange] = useState({
    startDate: subDays(new Date(), 7),
    endDate: new Date()
  });
  
  // Map reference
  const mapRef = useRef(null);
  
  // Fetch data function
  const fetchData = useCallback(async () => {
    try {
      setLoading(true);
      
      // In a real application, this would be an API call
      // const response = await axios.get(apiUrl, { params: { 
      //   wasteType: filters.wasteType,
      //   zone: filters.zone,
      //   fillLevelMin: filters.fillLevelRange[0],
      //   fillLevelMax: filters.fillLevelRange[1],
      //   startDate: dateRange.startDate ? format(dateRange.startDate, 'yyyy-MM-dd') : undefined,
      //   endDate: dateRange.endDate ? format(dateRange.endDate, 'yyyy-MM-dd') : undefined
      // }});
      // setBins(response.data);
      
      // For demo purposes, we'll generate mock data
      const mockBins = generateMockData(50);
      setBins(mockBins);
      
      // Mock vehicles data
      const mockVehicles = generateMockVehicles(3);
      setVehicles(mockVehicles);
      
      setLastUpdated(new Date());
      setError(null);
    } catch (err) {
      console.error("Error fetching waste bin data:", err);
      setError("Failed to load waste bin data. Please try again.");
    } finally {
      setLoading(false);
    }
  }, [apiUrl, filters, dateRange]);
  
  // Generate mock data function
  const generateMockData = (count) => {
    const wasteBins = [];
    
    // Base location with some variation
    const baseLat = initialCenter[0];
    const baseLng = initialCenter[1];
    
    for (let i = 1; i <= count; i++) {
      // Random location within ~2km of center
      const lat = baseLat + (Math.random() - 0.5) * 0.03;
      const lng = baseLng + (Math.random() - 0.5) * 0.03;
      
      // Random waste type
      const wasteType = WASTE_TYPES[Math.floor(Math.random() * WASTE_TYPES.length)].id;
      
      // Random zone
      const zone = COLLECTION_ZONES[Math.floor(Math.random() * COLLECTION_ZONES.length)].id;
      
      // Random fill level with bias toward lower values
      const fillLevel = Math.min(100, Math.floor(Math.random() * Math.random() * 120));
      
      // Random last collection date (between 1 and 10 days ago)
      const lastCollected = subDays(new Date(), Math.floor(Math.random() * 10) + 1);
      
      // Mock fill level history
      const history = [];
      let prevLevel = 0;
      for (let day = 7; day >= 0; day--) {
        // Generate increasing fill levels over time
        if (day === 0) {
          history.push({ date: 'now', level: fillLevel });
        } else {
          // Lower fill levels in the past
          const historyLevel = Math.max(0, fillLevel - (Math.random() * 10 * day));
          history.push({ date: `${day}d ago`, level: Math.round(historyLevel) });
        }
      }
      
      wasteBins.push({
        id: i,
        name: `Waste Bin #${i}`,
        lat,
        lng,
        wasteType,
        zone,
        fillLevel,
        capacity: '120 L',
        lastCollected,
        location: `Location ${i}`,
        history
      });
    }
    
    return wasteBins;
  };
  
  // Generate mock collection vehicles
  const generateMockVehicles = (count) => {
    const vehicles = [];
    
    // Base location with some variation
    const baseLat = initialCenter[0];
    const baseLng = initialCenter[1];
    
    for (let i = 1; i <= count; i++) {
      // Random location within ~2km of center
      const lat = baseLat + (Math.random() - 0.5) * 0.03;
      const lng = baseLng + (Math.random() - 0.5) * 0.03;
      
      // Random zone
      const zone = COLLECTION_ZONES[Math.floor(Math.random() * COLLECTION_ZONES.length)].id;
      
      // Random waste type
      const wasteType = WASTE_TYPES[Math.floor(Math.random() * WASTE_TYPES.length)].id;
      
      // Random fill level
      const fillLevel = Math.floor(Math.random() * 100);
      
      vehicles.push({
        id: `v-${i}`,
        name: `Collection Vehicle ${i}`,
        lat,
        lng,
        zone,
        wasteType,
        fillLevel,
        speed: Math.floor(Math.random() * 30) + 10, // 10-40 km/h
        status: Math.random() > 0.2 ? 'active' : 'idle',
        driver: `Driver ${i}`
      });
    }
    
    return vehicles;
  };
  
  // Initial data fetch
  useEffect(() => {
    fetchData();
    
    // Optional: Set up real-time updates
    const interval = setInterval(fetchData, 60000); // Update every minute
    
    return () => clearInterval(interval);
  }, [fetchData]);
  
  // Filter the bins based on current filters
  const filteredBins = useMemo(() => {
    return bins.filter(bin => {
      // Filter by waste type
      if (filters.wasteType && bin.wasteType !== filters.wasteType) {
        return false;
      }
      
      // Filter by zone
      if (filters.zone && bin.zone !== filters.zone) {
        return false;
      }
      
      // Filter by fill level range
      if (bin.fillLevel < filters.fillLevelRange[0] || bin.fillLevel > filters.fillLevelRange[1]) {
        return false;
      }
      
      return true;
    });
  }, [bins, filters]);
  
  // Create heatmap data points weighted by fill level
  const heatmapData = useMemo(() => {
    return filteredBins.map(bin => ({
      lat: bin.lat,
      lng: bin.lng, 
      intensity: bin.fillLevel / 25 // Scale intensity by fill level (higher fill = more intense)
    }));
  }, [filteredBins]);
  
  // Reset filters to default
  const resetFilters = () => {
    setFilters({
      wasteType: '',
      zone: '',
      fillLevelRange: [0, 100]
    });
    
    setDateRange({
      startDate: subDays(new Date(), 7),
      endDate: new Date()
    });
  };
  
  // Export data as CSV
  const exportData = () => {
    // Create CSV content
    const headers = ['ID', 'Name', 'Latitude', 'Longitude', 'Waste Type', 'Zone', 'Fill Level', 'Last Collected'];
    const csvRows = [
      headers.join(','),
      ...filteredBins.map(bin => 
        [
          bin.id,
          `"${bin.name}"`,
          bin.lat,
          bin.lng,
          bin.wasteType,
          bin.zone,
          bin.fillLevel,
          bin.lastCollected ? format(new Date(bin.lastCollected), 'yyyy-MM-dd') : 'Never'
        ].join(',')
      )
    ];
    
    const csvContent = csvRows.join('\n');
    
    // Create a blob and download
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    saveAs(blob, `waste-bins-export-${format(new Date(), 'yyyyMMdd')}.csv`);
  };
  
  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <AppBar position="static" color="default" elevation={0} sx={{ borderBottom: '1px solid #ddd' }}>
        <Toolbar>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1, display: 'flex', alignItems: 'center' }}>
            <MapIcon sx={{ mr: 1 }} />
            Waste Heatmap Dashboard
          </Typography>
          
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            {lastUpdated && (
              <Typography variant="body2" color="text.secondary" sx={{ mr: 2 }}>
                Last updated: {format(lastUpdated, 'MMM d, HH:mm:ss')}
              </Typography>
            )}
            
            <Tooltip title="Download Data">
              <IconButton onClick={exportData} color="inherit" size="small" sx={{ mr: 1 }}>
                <DownloadIcon />
              </IconButton>
            </Tooltip>
            
            <Tooltip title="Toggle Heatmap">
              <IconButton 
                onClick={() => setShowHeatmap(!showHeatmap)} 
                color={showHeatmap ? "primary" : "default"}
                size="small"
                sx={{ mr: 1 }}
              >
                <LayersIcon />
              </IconButton>
            </Tooltip>
            
            <Tooltip title="Toggle Legend">
              <IconButton 
                onClick={() => setMapLegendVisible(!mapLegendVisible)} 
                color={mapLegendVisible ? "primary" : "default"}
                size="small"
              >
                <InfoIcon />
              </IconButton>
            </Tooltip>
          </Box>
        </Toolbar>
        
        {/* Tabs for different views */}
        <Tabs 
          value={activeTab} 
          onChange={(e, newValue) => setActiveTab(newValue)} 
          indicatorColor="primary"
          textColor="primary"
          variant="fullWidth"
        >
          <Tab icon={<MapIcon />} label="Map View" id="waste-map-tab-0" />
          <Tab icon={<BarChartIcon />} label="Analytics" id="waste-map-tab-1" />
        </Tabs>
      </AppBar>
      
      {/* Error alert if needed */}
      {error && (
        <Alert severity="error" sx={{ m: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}
      
      {/* Map View Tab */}
      <TabPanel value={activeTab} index={0} sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column', p: 0 }}>
        <Box sx={{ p: 2 }}>
          <FilterControls
            filters={filters}
            setFilters={setFilters}
            dateRange={dateRange}
            setDateRange={setDateRange}
            onRefresh={fetchData}
            onReset={resetFilters}
          />
        </Box>
        
        <Box sx={{ flexGrow: 1, position: 'relative' }}>
          {/* Loading indicator */}
          {loading && (
            <CircularProgress 
              sx={{ 
                position: 'absolute', 
                top: '50%', 
                left: '50%', 
                transform: 'translate(-50%, -50%)',
                zIndex: 1000
              }} 
            />
          )}
          
          {/* Map container */}
          <MapContainer
            center={initialCenter}
            zoom={initialZoom}
            style={{ height: '100%', width: '100%' }}
            whenCreated={mapInstance => { mapRef.current = mapInstance; }}
            zoomControl={false}
          >
            <ZoomControl position="topright" />
            <ScaleControl position="bottomleft" />
            
            <LayersControl position="topright">
              {/* Base layers */}
              <LayersControl.BaseLayer checked name="OpenStreetMap">
                <TileLayer
                  url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                  attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                />
              </LayersControl.BaseLayer>
              
              <LayersControl.BaseLayer name="Satellite">
                <TileLayer
                  url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                  attribution='&copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'
                />
              </LayersControl.Base
