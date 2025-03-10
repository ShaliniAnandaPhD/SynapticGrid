/**
 * Energy Usage Dashboard
 * ---------------------
 * React-based visualization dashboard for monitoring and analyzing
 * energy demand, supply, and optimization strategies in a smart grid.
 * 
 * Provides interactive charts, real-time updates, and optimization insights.
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { 
  LineChart, Line, AreaChart, Area, BarChart, Bar, PieChart, Pie, 
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  ComposedChart, Scatter, Cell, RadialBarChart, RadialBar
} from 'recharts';
import { 
  Container, Grid, Paper, Typography, Box, Button, 
  FormControl, InputLabel, MenuItem, Select, Tab, Tabs,
  Table, TableBody, TableCell, TableContainer, TableHead, 
  TableRow, IconButton, Divider, useTheme, CircularProgress,
  Card, CardContent, CardHeader, Switch, FormControlLabel, Chip
} from '@mui/material';
import { 
  Refresh as RefreshIcon, 
  TrendingUp as TrendingUpIcon, 
  TrendingDown as TrendingDownIcon,
  Warning as WarningIcon,
  Check as CheckIcon,
  Save as SaveIcon,
  Lightbulb as LightbulbIcon,
  BatteryChargingFull as BatteryIcon,
  Speed as SpeedIcon,
  ShowChart as ShowChartIcon,
  DateRange as DateRangeIcon
} from '@mui/icons-material';
import { format, subDays, subHours, parseISO } from 'date-fns';
import axios from 'axios';

// Custom hook for fetching data with loading and error states
const useDataFetching = (url, initialData = [], interval = null) => {
  const [data, setData] = useState(initialData);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  const fetchData = useCallback(async () => {
    try {
      setLoading(true);
      // In a real app, this would be an actual API call
      // const response = await axios.get(url);
      // setData(response.data);
      
      // For demo purposes, we'll generate mock data
      const mockData = generateMockData(url);
      setData(mockData);
      setError(null);
    } catch (err) {
      console.error("Error fetching data:", err);
      setError(err.message || "Failed to fetch data");
    } finally {
      setLoading(false);
    }
  }, [url]);
  
  useEffect(() => {
    fetchData();
    
    // Set up interval for real-time updates if requested
    if (interval) {
      const timer = setInterval(fetchData, interval);
      return () => clearInterval(timer);
    }
  }, [fetchData, interval]);
  
  return { data, loading, error, refetch: fetchData };
};

// Mock data generation (would be replaced with real API calls)
const generateMockData = (dataType) => {
  // Energy production mix
  if (dataType === 'energyMix') {
    return [
      { name: 'Solar', value: 35, color: '#FFC107' },
      { name: 'Wind', value: 25, color: '#4CAF50' },
      { name: 'Battery', value: 15, color: '#2196F3' },
      { name: 'Grid', value: 20, color: '#9C27B0' },
      { name: 'Backup', value: 5, color: '#FF5722' }
    ];
  }
  
  // Energy consumption by type
  if (dataType === 'consumptionByType') {
    return [
      { name: 'Residential', value: 40, color: '#3F51B5' },
      { name: 'Commercial', value: 30, color: '#00BCD4' },
      { name: 'Industrial', value: 25, color: '#607D8B' },
      { name: 'Public', value: 5, color: '#8BC34A' }
    ];
  }
  
  // Energy usage trend (last 24 hours)
  if (dataType === 'usageTrend') {
    const hours = 24;
    return Array(hours).fill().map((_, i) => {
      const hour = 23 - i;
      const timestamp = format(subHours(new Date(), i), 'HH:mm');
      const baseProduction = 250 + Math.sin(hour / 3) * 50;
      const baseConsumption = 200 + 100 * Math.sin((hour - 5) * Math.PI / 12);
      
      // Add some random variation
      const production = Math.max(0, baseProduction + (Math.random() * 40 - 20));
      const consumption = Math.max(0, baseConsumption + (Math.random() * 30 - 15));
      const surplus = production - consumption;
      
      return {
        timestamp,
        hour: hour,
        production: parseInt(production),
        consumption: parseInt(consumption),
        surplus: parseInt(surplus)
      };
    }).reverse();
  }
  
  // Energy optimization recommendations
  if (dataType === 'optimizationRecommendations') {
    return [
      {
        id: 1,
        title: 'Shift Non-Essential Consumption',
        description: 'Move 15% of non-essential energy usage to off-peak hours for better grid efficiency.',
        impact: 'High',
        savingsEstimate: '12-15%',
        status: 'New'
      },
      {
        id: 2,
        title: 'Increase Solar Utilization',
        description: 'Current solar production is below capacity. Check for panel obstructions or maintenance needs.',
        impact: 'Medium',
        savingsEstimate: '5-8%',
        status: 'In Progress'
      },
      {
        id: 3,
        title: 'Battery Charging Strategy',
        description: 'Optimize battery charging to match forecast weather conditions. Expected cloudy period tomorrow.',
        impact: 'Medium',
        savingsEstimate: '7-10%',
        status: 'Implemented'
      },
      {
        id: 4,
        title: 'Demand Response Program',
        description: 'Enroll eligible devices in automated demand response to reduce peak load.',
        impact: 'High',
        savingsEstimate: '10-20%',
        status: 'New'
      }
    ];
  }
  
  // Peak demand events
  if (dataType === 'peakEvents') {
    return [
      {
        id: 1,
        startTime: '08:30',
        endTime: '10:15',
        date: format(subDays(new Date(), 1), 'yyyy-MM-dd'),
        peakValue: 342,
        duration: 105,
        category: 'Morning Peak'
      },
      {
        id: 2,
        startTime: '18:45',
        endTime: '20:30',
        date: format(subDays(new Date(), 1), 'yyyy-MM-dd'),
        peakValue: 385,
        duration: 105,
        category: 'Evening Peak'
      },
      {
        id: 3,
        startTime: '12:15',
        endTime: '13:30',
        date: format(new Date(), 'yyyy-MM-dd'),
        peakValue: 315,
        duration: 75,
        category: 'Midday Peak'
      }
    ];
  }
  
  // Monthly usage comparison
  if (dataType === 'monthlyComparison') {
    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    const currentMonth = new Date().getMonth();
    
    return months.map((month, index) => {
      // Create a seasonal pattern with random variation
      const seasonalFactor = 1 - 0.3 * Math.cos((index - 6) * Math.PI / 6);
      const baseValue = 300 * seasonalFactor;
      const randomVariation = Math.random() * 40 - 20;
      
      // Current month is incomplete and usually lower
      const actualValue = index === currentMonth ? 
        baseValue * 0.7 + randomVariation : 
        baseValue + randomVariation;
        
      // Efficiency scores increase over time (learning)
      const baseline = baseValue * 1.1 + (Math.random() * 30 - 15);
      const efficiency = Math.min(98, 80 + index * 1.5 + (Math.random() * 5 - 2.5));
      
      return {
        month: month,
        actual: parseInt(actualValue),
        baseline: parseInt(baseline),
        efficiency: parseInt(efficiency)
      };
    });
  }
  
  // Default empty data
  return [];
};

// TabPanel component for tab content
const TabPanel = ({ children, value, index, ...other }) => (
  <div
    role="tabpanel"
    hidden={value !== index}
    id={`energy-tabpanel-${index}`}
    aria-labelledby={`energy-tab-${index}`}
    {...other}
  >
    {value === index && <Box p={3}>{children}</Box>}
  </div>
);

// Helper to create tab accessibility props
const a11yProps = (index) => ({
  id: `energy-tab-${index}`,
  'aria-controls': `energy-tabpanel-${index}`,
});

// Formatter for kWh values
const formatKWh = (value) => `${value} kWh`;

// Color palette for consistent styling
const COLORS = {
  production: '#4CAF50',
  consumption: '#F44336',
  surplus: '#2196F3',
  deficit: '#FF9800',
  solar: '#FFC107',
  wind: '#81C784',
  battery: '#64B5F6',
  grid: '#9575CD',
  background: '#f5f5f5',
  text: '#333333',
  border: '#dddddd'
};

/**
 * Main Energy Dashboard Component
 */
const EnergyDashboard = () => {
  const theme = useTheme();
  
  // State
  const [tabValue, setTabValue] = useState(0);
  const [dateRange, setDateRange] = useState('24h');
  const [realTime, setRealTime] = useState(true);
  const [selectedView, setSelectedView] = useState('overview');
  
  // Fetch data
  const { 
    data: energyMix, 
    loading: loadingMix 
  } = useDataFetching('energyMix');
  
  const { 
    data: consumptionByType, 
    loading: loadingConsumption 
  } = useDataFetching('consumptionByType');
  
  const { 
    data: usageTrend, 
    loading: loadingTrend, 
    refetch: refetchTrend 
  } = useDataFetching('usageTrend', [], realTime ? 10000 : null);
  
  const {
    data: monthlyComparison,
    loading: loadingMonthly
  } = useDataFetching('monthlyComparison');
  
  const {
    data: optimizationRecommendations,
    loading: loadingRecommendations
  } = useDataFetching('optimizationRecommendations');
  
  const {
    data: peakEvents,
    loading: loadingPeaks
  } = useDataFetching('peakEvents');
  
  // Derived stats
  const currentStats = useMemo(() => {
    if (!usageTrend || usageTrend.length === 0) return null;
    
    const latest = usageTrend[usageTrend.length - 1];
    const previous = usageTrend[usageTrend.length - 2];
    
    // Calculate change percentages
    const productionChange = previous ? 
      ((latest.production - previous.production) / previous.production * 100).toFixed(1) : 0;
    
    const consumptionChange = previous ? 
      ((latest.consumption - previous.consumption) / previous.consumption * 100).toFixed(1) : 0;
    
    const surplusChange = previous && previous.surplus ? 
      ((latest.surplus - previous.surplus) / Math.abs(previous.surplus) * 100).toFixed(1) : 0;
    
    return {
      timestamp: latest.timestamp,
      production: latest.production,
      consumption: latest.consumption,
      surplus: latest.surplus,
      productionChange: parseFloat(productionChange),
      consumptionChange: parseFloat(consumptionChange),
      surplusChange: parseFloat(surplusChange)
    };
  }, [usageTrend]);
  
  // Total energy mix
  const totalEnergyProduction = useMemo(() => {
    return energyMix.reduce((sum, source) => sum + source.value, 0);
  }, [energyMix]);
  
  // Handle tab change
  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };
  
  // Handle date range change
  const handleDateRangeChange = (event) => {
    setDateRange(event.target.value);
  };
  
  // Handle real-time toggle
  const handleRealTimeToggle = (event) => {
    setRealTime(event.target.checked);
  };
  
  // Format impact as chip
  const getImpactChip = (impact) => {
    let color = 'default';
    if (impact === 'High') color = 'error';
    if (impact === 'Medium') color = 'warning';
    if (impact === 'Low') color = 'info';
    
    return <Chip label={impact} color={color} size="small" />;
  };
  
  // Format status as chip
  const getStatusChip = (status) => {
    let color = 'default';
    if (status === 'New') color = 'info';
    if (status === 'In Progress') color = 'warning';
    if (status === 'Implemented') color = 'success';
    
    return <Chip label={status} color={color} size="small" />;
  };
  
  return (
    <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
      {/* Dashboard Header */}
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="h4" component="h1" gutterBottom>
          <ShowChartIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
          Energy Usage Dashboard
        </Typography>
        
        <Box>
          <FormControlLabel
            control={
              <Switch
                checked={realTime}
                onChange={handleRealTimeToggle}
                color="primary"
              />
            }
            label="Real-time Updates"
          />
          
          <FormControl variant="outlined" size="small" sx={{ ml: 2, minWidth: 120 }}>
            <InputLabel id="date-range-label">Date Range</InputLabel>
            <Select
              labelId="date-range-label"
              value={dateRange}
              onChange={handleDateRangeChange}
              label="Date Range"
              startAdornment={<DateRangeIcon color="action" fontSize="small" sx={{ mr: 1 }} />}
            >
              <MenuItem value="24h">Last 24 Hours</MenuItem>
              <MenuItem value="7d">Last 7 Days</MenuItem>
              <MenuItem value="30d">Last 30 Days</MenuItem>
              <MenuItem value="90d">Last 90 Days</MenuItem>
            </Select>
          </FormControl>
          
          <IconButton color="primary" onClick={refetchTrend} sx={{ ml: 1 }}>
            <RefreshIcon />
          </IconButton>
        </Box>
      </Box>
      
      {/* Current Stats Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        {/* Production Card */}
        <Grid item xs={12} md={4}>
          <Card elevation={2}>
            <CardContent>
              <Typography variant="h6" color="textSecondary" gutterBottom>
                <TrendingUpIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                Current Production
              </Typography>
              
              {loadingTrend ? (
                <CircularProgress size={24} />
              ) : (
                <>
                  <Typography variant="h4" component="div" sx={{ mb: 1 }}>
                    {currentStats?.production || 0} kWh
                  </Typography>
                  
                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                    {currentStats?.productionChange > 0 ? (
                      <TrendingUpIcon sx={{ color: 'success.main', mr: 0.5 }} />
                    ) : (
                      <TrendingDownIcon sx={{ color: 'error.main', mr: 0.5 }} />
                    )}
                    
                    <Typography
                      variant="body2"
                      color={currentStats?.productionChange > 0 ? 'success.main' : 'error.main'}
                    >
                      {currentStats?.productionChange > 0 ? '+' : ''}
                      {currentStats?.productionChange}% from previous hour
                    </Typography>
                  </Box>
                </>
              )}
            </CardContent>
          </Card>
        </Grid>
        
        {/* Consumption Card */}
        <Grid item xs={12} md={4}>
          <Card elevation={2}>
            <CardContent>
              <Typography variant="h6" color="textSecondary" gutterBottom>
                <TrendingDownIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                Current Consumption
              </Typography>
              
              {loadingTrend ? (
                <CircularProgress size={24} />
              ) : (
                <>
                  <Typography variant="h4" component="div" sx={{ mb: 1 }}>
                    {currentStats?.consumption || 0} kWh
                  </Typography>
                  
                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                    {currentStats?.consumptionChange < 0 ? (
                      <TrendingDownIcon sx={{ color: 'success.main', mr: 0.5 }} />
                    ) : (
                      <TrendingUpIcon sx={{ color: 'error.main', mr: 0.5 }} />
                    )}
                    
                    <Typography
                      variant="body2"
                      color={currentStats?.consumptionChange < 0 ? 'success.main' : 'error.main'}
                    >
                      {currentStats?.consumptionChange > 0 ? '+' : ''}
                      {currentStats?.consumptionChange}% from previous hour
                    </Typography>
                  </Box>
                </>
              )}
            </CardContent>
          </Card>
        </Grid>
        
        {/* Energy Balance Card */}
        <Grid item xs={12} md={4}>
          <Card elevation={2}>
            <CardContent>
              <Typography variant="h6" color="textSecondary" gutterBottom>
                <BatteryIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                Energy Balance
              </Typography>
              
              {loadingTrend ? (
                <CircularProgress size={24} />
              ) : (
                <>
                  <Typography 
                    variant="h4" 
                    component="div" 
                    sx={{ mb: 1 }}
                    color={currentStats?.surplus >= 0 ? 'success.main' : 'error.main'}
                  >
                    {currentStats?.surplus >= 0 ? '+' : ''}
                    {currentStats?.surplus || 0} kWh
                  </Typography>
                  
                  <Typography variant="body2" color="textSecondary">
                    {currentStats?.surplus >= 0 
                      ? `Surplus energy available for storage or grid`
                      : `Deficit requiring additional power sources`}
                  </Typography>
                </>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
      
      {/* Tabs for different sections */}
      <Paper elevation={3} sx={{ mb: 4 }}>
        <Tabs 
          value={tabValue} 
          onChange={handleTabChange} 
          indicatorColor="primary"
          textColor="primary"
          variant="fullWidth"
        >
          <Tab label="Usage Trends" icon={<ShowChartIcon />} {...a11yProps(0)} />
          <Tab label="Energy Sources" icon={<LightbulbIcon />} {...a11yProps(1)} />
          <Tab label="Optimization" icon={<SpeedIcon />} {...a11yProps(2)} />
        </Tabs>
        
        {/* Usage Trends Tab */}
        <TabPanel value={tabValue} index={0}>
          <Grid container spacing={4}>
            {/* Energy Usage Trend Chart */}
            <Grid item xs={12}>
              <Paper elevation={2} sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>Energy Usage Trend (Last 24 Hours)</Typography>
                
                {loadingTrend ? (
                  <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
                    <CircularProgress />
                  </Box>
                ) : (
                  <ResponsiveContainer width="100%" height={400}>
                    <ComposedChart data={usageTrend}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="timestamp" />
                      <YAxis yAxisId="left" orientation="left" label={{ value: 'Energy (kWh)', angle: -90, position: 'insideLeft' }} />
                      <YAxis yAxisId="right" orientation="right" label={{ value: 'Surplus/Deficit (kWh)', angle: 90, position: 'insideRight' }} />
                      <Tooltip formatter={(value) => `${value} kWh`} />
                      <Legend />
                      
                      <Area 
                        yAxisId="left"
                        type="monotone" 
                        dataKey="production" 
                        name="Production" 
                        fill={COLORS.production} 
                        stroke={COLORS.production}
                        fillOpacity={0.3}
                      />
                      <Area 
                        yAxisId="left"
                        type="monotone" 
                        dataKey="consumption" 
                        name="Consumption" 
                        fill={COLORS.consumption} 
                        stroke={COLORS.consumption}
                        fillOpacity={0.3}
                      />
                      <Line 
                        yAxisId="right"
                        type="monotone" 
                        dataKey="surplus" 
                        name="Energy Balance" 
                        stroke={COLORS.surplus}
                        dot={true}
                        strokeWidth={2}
                      />
                    </ComposedChart>
                  </ResponsiveContainer>
                )}
              </Paper>
            </Grid>
            
            {/* Peak Demand Events */}
            <Grid item xs={12} md={6}>
              <Paper elevation={2} sx={{ p: 3, height: '100%' }}>
                <Typography variant="h6" gutterBottom>Recent Peak Demand Events</Typography>
                
                {loadingPeaks ? (
                  <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
                    <CircularProgress />
                  </Box>
                ) : (
                  <TableContainer>
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell>Date</TableCell>
                          <TableCell>Time</TableCell>
                          <TableCell>Duration</TableCell>
                          <TableCell>Peak Value</TableCell>
                          <TableCell>Category</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {peakEvents.map((event) => (
                          <TableRow key={event.id} hover>
                            <TableCell>{event.date}</TableCell>
                            <TableCell>{event.startTime} - {event.endTime}</TableCell>
                            <TableCell>{event.duration} min</TableCell>
                            <TableCell>{event.peakValue} kWh</TableCell>
                            <TableCell>
                              <Chip 
                                label={event.category} 
                                size="small" 
                                color="primary" 
                                variant="outlined" 
                              />
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                )}
              </Paper>
            </Grid>
            
            {/* Monthly Comparison */}
            <Grid item xs={12} md={6}>
              <Paper elevation={2} sx={{ p: 3, height: '100%' }}>
                <Typography variant="h6" gutterBottom>Monthly Usage Comparison</Typography>
                
                {loadingMonthly ? (
                  <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
                    <CircularProgress />
                  </Box>
                ) : (
                  <ResponsiveContainer width="100%" height={300}>
                    <ComposedChart data={monthlyComparison}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="month" />
                      <YAxis yAxisId="left" orientation="left" label={{ value: 'Energy (kWh)', angle: -90, position: 'insideLeft' }} />
                      <YAxis yAxisId="right" orientation="right" domain={[0, 100]} label={{ value: 'Efficiency (%)', angle: 90, position: 'insideRight' }} />
                      <Tooltip formatter={(value, name) => name === 'efficiency' ? `${value}%` : `${value} kWh`} />
                      <Legend />
                      
                      <Bar 
                        yAxisId="left"
                        dataKey="actual" 
                        name="Actual Usage" 
                        fill={COLORS.consumption} 
                        barSize={20}
                      />
                      <Bar 
                        yAxisId="left"
                        dataKey="baseline" 
                        name="Baseline Projection" 
                        fill="#9E9E9E" 
                        barSize={20}
                        fillOpacity={0.4}
                      />
                      <Line 
                        yAxisId="right"
                        type="monotone" 
                        dataKey="efficiency" 
                        name="Energy Efficiency" 
                        stroke="#FF5722"
                        strokeWidth={2}
                      />
                    </ComposedChart>
                  </ResponsiveContainer>
                )}
              </Paper>
            </Grid>
          </Grid>
        </TabPanel>
        
        {/* Energy Sources Tab */}
        <TabPanel value={tabValue} index={1}>
          <Grid container spacing={4}>
            {/* Energy Mix */}
            <Grid item xs={12} md={6}>
              <Paper elevation={2} sx={{ p: 3, height: '100%' }}>
                <Typography variant="h6" gutterBottom>Energy Production Mix</Typography>
                
                {loadingMix ? (
                  <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
                    <CircularProgress />
                  </Box>
                ) : (
                  <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                    <ResponsiveContainer width="100%" height={300}>
                      <PieChart>
                        <Pie
                          data={energyMix}
                          dataKey="value"
                          nameKey="name"
                          cx="50%"
                          cy="50%"
                          outerRadius={100}
                          labelLine={true}
                          label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                        >
                          {energyMix.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.color} />
                          ))}
                        </Pie>
                        <Tooltip formatter={(value) => `${value} kWh (${((value / totalEnergyProduction) * 100).toFixed(1)}%)`} />
                      </PieChart>
                    </ResponsiveContainer>
                    
                    <Typography variant="body2" color="textSecondary" align="center">
                      Total Production: {totalEnergyProduction} kWh
                    </Typography>
                    
                    <Box sx={{ mt: 2, width: '100%' }}>
                      <Typography variant="subtitle2" gutterBottom>Energy Source Details</Typography>
                      <TableContainer>
                        <Table size="small">
                          <TableHead>
                            <TableRow>
                              <TableCell>Source</TableCell>
                              <TableCell align="right">Production (kWh)</TableCell>
                              <TableCell align="right">Percentage</TableCell>
                            </TableRow>
                          </TableHead>
                          <TableBody>
                            {energyMix.map((source) => (
                              <TableRow key={source.name} hover>
                                <TableCell>
                                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                                    <Box 
                                      sx={{ 
                                        width: 12, 
                                        height: 12, 
                                        borderRadius: '50%', 
                                        backgroundColor: source.color,
                                        mr: 1 
                                      }} 
                                    />
                                    {source.name}
                                  </Box>
                                </TableCell>
                                <TableCell align="right">{source.value}</TableCell>
                                <TableCell align="right">
                                  {((source.value / totalEnergyProduction) * 100).toFixed(1)}%
                                </TableCell>
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      </TableContainer>
                    </Box>
                  </Box>
                )}
              </Paper>
            </Grid>
            
            {/* Consumption By Type */}
            <Grid item xs={12} md={6}>
              <Paper elevation={2} sx={{ p: 3, height: '100%' }}>
                <Typography variant="h6" gutterBottom>Energy Consumption By Type</Typography>
                
                {loadingConsumption ? (
                  <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
                    <CircularProgress />
                  </Box>
                ) : (
                  <ResponsiveContainer width="100%" height={400}>
                    <BarChart 
                      data={consumptionByType}
                      layout="vertical"
                      margin={{ top: 5, right: 30, left: 100, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis type="number" label={{ value: 'Energy (kWh)', position: 'insideBottom', offset: -5 }} />
                      <YAxis type="category" dataKey="name" />
                      <Tooltip formatter={(value) => `${value} kWh`} />
                      <Legend />
                      <Bar dataKey="value" name="Consumption">
                        {consumptionByType.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                )}
              </Paper>
            </Grid>
            
            {/* Energy Distribution */}
            <Grid item xs={12}>
              <Paper elevation={2} sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>Daily Energy Distribution Pattern</Typography>
                
                {loadingTrend ? (
                  <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
                    <CircularProgress />
                  </Box>
                ) : (
                  <ResponsiveContainer width="100%" height={350}>
                    <AreaChart data={usageTrend}>
                      <defs>
                        <linearGradient id="colorProduction" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor={COLORS.production} stopOpacity={0.8}/>
                          <stop offset="95%" stopColor={COLORS.production} stopOpacity={0.1}/>
                        </linearGradient>
                        <linearGradient id="colorConsumption" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor={COLORS.consumption} stopOpacity={0.8}/>
                          <stop offset="95%" stopColor={COLORS.consumption} stopOpacity={0.1}/>
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="timestamp" />
                      <YAxis label={{ value: 'Energy (kWh)', angle: -90, position: 'insideLeft' }} />
                      <Tooltip formatter={(value) => `${value} kWh`} />
                      <Legend />
                      <Area 
                        type="monotone" 
                        dataKey="production" 
                        name="Production" 
                        stroke={COLORS.production} 
                        fillOpacity={1} 
                        fill="url(#colorProduction)" 
                      />
                      <Area 
                        type="monotone" 
                        dataKey="consumption" 
                        name="Consumption" 
                        stroke={COLORS.consumption} 
                        fillOpacity={1} 
                        fill="url(#colorConsumption)" 
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                )}
                
                <Box sx={{ mt: 2 }}>
                  <Typography variant="body2" color="textSecondary">
                    This chart shows the distribution of energy production and consumption throughout the day,
                    helping identify patterns and optimization opportunities.
                  </Typography>
                </Box>
              </Paper>
            </Grid>
          </Grid>
        </TabPanel>
        
        {/* Optimization Tab */}
        <TabPanel value={tabValue} index={2}>
          <Grid container spacing={4}>
            {/* Optimization Recommendations */}
            <Grid item xs={12}>
              <Paper elevation={2} sx={{ p: 3 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                  <Typography variant="h6">Energy Optimization Recommendations</Typography>
                  <Button 
                    variant="outlined" 
                    color="primary" 
                    startIcon={<SaveIcon />}
                    size="small"
                  >
                    Export Recommendations
                  </Button>
                </Box>
                
                {loadingRecommendations ? (
                  <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
                    <CircularProgress />
                  </Box>
                ) : (
                  <TableContainer>
                    <Table>
                      <TableHead>
                        <TableRow>
                          <TableCell>Recommendation</TableCell>
                          <TableCell>Description</TableCell>
                          <TableCell>Impact</TableCell>
                          <TableCell>Est. Savings</TableCell>
                          <TableCell>Status</TableCell>
                          <TableCell>Actions</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {optimizationRecommendations.map((rec) => (
                          <TableRow key={rec.id} hover>
                            <TableCell><Typography variant="subtitle2">{rec.title}</Typography></TableCell>
                            <TableCell>{rec.description}</TableCell>
                            <TableCell>{getImpactChip(rec.impact)}</TableCell>
                            <TableCell>{rec.savingsEstimate}</TableCell>
                            <TableCell>{getStatusChip(rec.status)}</TableCell>
                            <TableCell>
                              <Button size="small" color="primary">Details</Button>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                )}
              </Paper>
            </Grid>
            
            {/* Optimization Impact */}
            <Grid item xs={12} md={6}>
              <Paper elevation={2} sx={{ p: 3, height: '100%' }}>
                <Typography variant="h6" gutterBottom>Optimization Impact Analysis</Typography>
                
                <ResponsiveContainer width="100%" height={300}>
                  <RadialBarChart 
                    cx="50%" 
                    cy="50%" 
                    innerRadius="20%" 
                    outerRadius="90%" 
                    barSize={20} 
                    data={[
                      { name: 'Cost Savings', value: 68, fill: '#FF9800' },
                      { name: 'Carbon Reduction', value: 78, fill: '#4CAF50' },
                      { name: 'Peak Load Reduction', value: 53, fill: '#2196F3' },
                      { name: 'Grid Stability', value: 82, fill: '#9C27B0' },
                    ]}
                  >
                    <RadialBar
                      minAngle={15}
                      background
                      clockWise
                      dataKey="value"
                      cornerRadius={10}
                      label={{ fill: '#666', position: 'insideStart' }}
                    />
                    <Legend 
                      iconSize={10} 
                      layout="vertical" 
                      verticalAlign="middle" 
                      align="right" 
                    />
                    <Tooltip formatter={(value) => `${value}%`} />
                  </RadialBarChart>
                </ResponsiveContainer>
                
                <Typography variant="body2" color="textSecondary" align="center" sx={{ mt: 2 }}>
                  Percentage improvement in each category due to implemented optimizations
                </Typography>
              </Paper>
            </Grid>
            
            {/* Energy Efficiency Targets */}
            <Grid item xs={12} md={6}>
              <Paper elevation={2} sx={{ p: 3, height: '100%' }}>
                <Typography variant="h6" gutterBottom>Energy Efficiency Targets</Typography>
                
                <TableContainer>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Metric</TableCell>
                        <TableCell>Current</TableCell>
                        <TableCell>Target</TableCell>
                        <TableCell>Progress</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      <TableRow hover>
                        <TableCell>Peak Load Reduction</TableCell>
                        <TableCell>15%</TableCell>
                        <TableCell>20%</TableCell>
                        <TableCell>
                          <Box sx={{ display: 'flex', alignItems: 'center' }}>
                            <Box sx={{ width: '100%', mr: 1 }}>
                              <LinearProgress variant="determinate" value={75} />
                            </Box>
                            <Box sx={{ minWidth: 35 }}>
                              <Typography variant="body2" color="textSecondary">75%</Typography>
                            </Box>
                          </Box>
                        </TableCell>
                      </TableRow>
                      <TableRow hover>
                        <TableCell>Carbon Emissions</TableCell>
                        <TableCell>-230kg</TableCell>
                        <TableCell>-300kg</TableCell>
                        <TableCell>
                          <Box sx={{ display: 'flex', alignItems: 'center' }}>
                            <Box sx={{ width: '100%', mr: 1 }}>
                              <LinearProgress variant="determinate" value={77} />
                            </Box>
                            <Box sx={{ minWidth: 35 }}>
                              <Typography variant="body2" color="textSecondary">77%</Typography>
                            </Box>
                          </Box>
                        </TableCell>
                      </TableRow>
                      <TableRow hover>
                        <TableCell>Self-Sufficiency</TableCell>
                        <TableCell>68%</TableCell>
                        <TableCell>75%</TableCell>
                        <TableCell>
                          <Box sx={{ display: 'flex', alignItems: 'center' }}>
                            <Box sx={{ width: '100%', mr: 1 }}>
                              <LinearProgress variant="determinate" value={91} />
                            </Box>
                            <Box sx={{ minWidth: 35 }}>
                              <Typography variant="body2" color="textSecondary">91%</Typography>
                            </Box>
                          </Box>
                        </TableCell>
                      </TableRow>
                      <TableRow hover>
                        <TableCell>Battery Utilization</TableCell>
                        <TableCell>45%</TableCell>
                        <TableCell>65%</TableCell>
                        <TableCell>
                          <Box sx={{ display: 'flex', alignItems: 'center' }}>
                            <Box sx={{ width: '100%', mr: 1 }}>
                              <LinearProgress variant="determinate" value={69} />
                            </Box>
                            <Box sx={{ minWidth: 35 }}>
                              <Typography variant="body2" color="textSecondary">69%</Typography>
                            </Box>
                          </Box>
                        </TableCell>
                      </TableRow>
                      <TableRow hover>
                        <TableCell>Cost Reduction</TableCell>
                        <TableCell>22%</TableCell>
                        <TableCell>30%</TableCell>
                        <TableCell>
                          <Box sx={{ display: 'flex', alignItems: 'center' }}>
                            <Box sx={{ width: '100%', mr: 1 }}>
                              <LinearProgress variant="determinate" value={73} />
                            </Box>
                            <Box sx={{ minWidth: 35 }}>
                              <Typography variant="body2" color="textSecondary">73%</Typography>
                            </Box>
                          </Box>
                        </TableCell>
                      </TableRow>
                    </TableBody>
                  </Table>
                </TableContainer>
                
                <Box sx={{ mt: 2, p: 2, backgroundColor: 'rgba(33, 150, 243, 0.1)', borderRadius: 1 }}>
                  <Typography variant="body2">
                    <strong>Next Review:</strong> All targets will be reassessed on 15th of next month based on
                    performance data and optimization strategy effectiveness.
                  </Typography>
                </Box>
              </Paper>
            </Grid>
          </Grid>
        </TabPanel>
      </Paper>
      
      {/* Footer with last updated info */}
      <Box sx={{ mt: 4, pt: 2, borderTop: `1px solid ${COLORS.border}`, display: 'flex', justifyContent: 'space-between' }}>
        <Typography variant="body2" color="textSecondary">
          Last updated: {new Date().toLocaleString()}
          {realTime && ' (real-time updates enabled)'}
        </Typography>
        
        <Typography variant="body2" color="textSecondary">
          Data source: Smart Grid Monitoring System
        </Typography>
      </Box>
    </Container>
  );
};

export default EnergyDashboard;

/**
 * SUMMARY:
 * --------
 * This component implements a comprehensive energy usage dashboard for monitoring
 * and analyzing energy patterns in a smart grid system. The dashboard provides:
 * 
 * - Real-time monitoring of energy production, consumption, and balance
 * - Visualization of energy mix and consumption patterns
 * - Historical trends and monthly comparisons
 * - Optimization recommendations and impact analysis
 * - Peak demand event tracking
 * 
 * The dashboard uses React with Material-UI for the interface and Recharts for
 * data visualization. It includes responsive design for different screen sizes
 * and simulated real-time updates.
 * 
 * TODO:
 * -----
 * - Implement actual API integration for real data
 * - Add user authentication and personalized views
 * - Create export functionality for reports
 * - Add predictive analytics for future energy usage
 * - Implement alert notifications for critical events
 * - Add geographic visualization of energy distribution
 */
