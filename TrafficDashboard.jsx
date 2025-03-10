import React, { useState, useEffect, useRef } from 'react';
import { LineChart, Line, BarChart, Bar, PieChart, Pie, AreaChart, Area, 
         XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { MapPin, AlertTriangle, Clock, ArrowUp, ArrowDown, Car, Activity, 
         Calendar, RefreshCw, Layers, Sliders, Download } from 'lucide-react';
import axios from 'axios';

/**
 * Traffic Dashboard - Real-time visualization of city traffic patterns
 * 
 * This dashboard connects to our traffic monitoring API endpoints and displays
 * real-time data about congestion, traffic flow, incidents, and optimization stats.
 * It's designed for both traffic management teams and for public displays in the
 * city's traffic management center.
 */

// API endpoints
const API_BASE_URL = process.env.REACT_APP_API_URL || 'https://api.smartcity.gov/traffic';
const REFRESH_INTERVAL = 60000; // 1 minute refresh

// Traffic status indicators
const STATUS_COLORS = {
  low: '#4caf50',       // Green
  moderate: '#ff9800',  // Orange
  heavy: '#f44336',     // Red
  incident: '#9c27b0',  // Purple
  closure: '#000000'    // Black
};

// Charts color palette
const CHART_COLORS = ['#2196f3', '#4caf50', '#f44336', '#ff9800', '#9c27b0', '#795548', '#607d8b'];

// Initial dummy data (fallback if API isn't available)
const FALLBACK_DATA = {
  congestionIndex: 34,
  trafficStatus: 'moderate',
  incidentCount: 3,
  lastUpdated: new Date().toISOString(),
  trafficFlow: [
    { time: '06:00', flow: 1200, congestion: 15 },
    { time: '07:00', flow: 2800, congestion: 34 },
    { time: '08:00', flow: 3500, congestion: 65 },
    { time: '09:00', flow: 3000, congestion: 48 },
    { time: '10:00', flow: 2200, congestion: 30 },
    { time: '11:00', flow: 1800, congestion: 25 },
    { time: '12:00', flow: 2100, congestion: 32 },
  ],
  congestionHotspots: [
    { location: 'Downtown Main St', severity: 75, duration: '45min' },
    { location: 'Highway 101 North', severity: 68, duration: '30min' },
    { location: 'Central Ave & Oak St', severity: 62, duration: '20min' },
    { location: 'Westside Bridge', severity: 54, duration: '15min' },
    { location: 'South Industrial Park', severity: 43, duration: '10min' },
  ],
  trafficIncidents: [
    { type: 'Accident', location: 'Highway 101 & 5th Ave', time: '08:32', status: 'Active', severity: 'Major' },
    { type: 'Construction', location: 'Main St between 1st-3rd', time: '07:00', status: 'Active', severity: 'Moderate' },
    { type: 'Disabled Vehicle', location: 'Westbound Bridge', time: '08:45', status: 'Clearing', severity: 'Minor' },
  ],
  modeDistribution: [
    { name: 'Private Cars', value: 65 },
    { name: 'Public Transit', value: 18 },
    { name: 'Cycling', value: 7 },
    { name: 'Walking', value: 8 },
    { name: 'Other', value: 2 },
  ],
  optimizationStats: {
    trafficLightOptimization: 68,
    routeSuggestionAcceptance: 42,
    congestionReduction: 23,
    fuelSavings: 18,
    emissionReduction: 15
  }
};

// Main dashboard component
const TrafficDashboard = () => {
  // State
  const [dashboardData, setDashboardData] = useState(FALLBACK_DATA);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [timeRange, setTimeRange] = useState('day'); // 'day', 'week', 'month'
  const [showSettings, setShowSettings] = useState(false);
  const [areaFilter, setAreaFilter] = useState('all');
  
  // Refs
  const refreshTimerRef = useRef(null);
  
  // Data fetching function
  const fetchDashboardData = async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${API_BASE_URL}/dashboard?timeRange=${timeRange}&area=${areaFilter}`);
      setDashboardData(response.data);
      setError(null);
    } catch (err) {
      console.error('Error fetching traffic data:', err);
      setError('Failed to load traffic data. Using cached data instead.');
      // We don't overwrite existing data on error to keep showing the last successful data
    } finally {
      setLoading(false);
    }
  };
  
  // On mount effect
  useEffect(() => {
    fetchDashboardData();
    
    // Setup refresh interval
    refreshTimerRef.current = setInterval(fetchDashboardData, REFRESH_INTERVAL);
    
    // Cleanup
    return () => {
      if (refreshTimerRef.current) {
        clearInterval(refreshTimerRef.current);
      }
    };
  }, [timeRange, areaFilter]);
  
  // Format date/time
  const formatDateTime = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleString();
  };
  
  // Export/download dashboard data
  const exportData = () => {
    const dataStr = JSON.stringify(dashboardData, null, 2);
    const dataUri = `data:application/json;charset=utf-8,${encodeURIComponent(dataStr)}`;
    
    const exportName = `traffic_data_${new Date().toISOString().split('T')[0]}`;
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', `${exportName}.json`);
    linkElement.click();
  };
  
  // Render loading or error state
  if (loading && !dashboardData) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center">
          <RefreshCw className="animate-spin h-12 w-12 text-blue-500 mx-auto" />
          <p className="mt-4 text-lg">Loading traffic data...</p>
        </div>
      </div>
    );
  }
  
  // Helper function to get traffic status color
  const getStatusColor = (status) => STATUS_COLORS[status] || STATUS_COLORS.moderate;
  
  // Helper function to get change indicator
  const getChangeIndicator = (value) => {
    if (value > 0) return <ArrowUp className="text-red-500" />;
    if (value < 0) return <ArrowDown className="text-green-500" />;
    return null;
  };
  
  return (
    <div className="min-h-screen bg-gray-100 p-4">
      {/* Top navigation bar */}
      <header className="bg-white p-4 rounded-lg shadow-md flex justify-between items-center mb-6">
        <div className="flex items-center">
          <Car className="h-8 w-8 text-blue-600 mr-2" />
          <h1 className="text-2xl font-bold text-gray-800">Smart City Traffic Dashboard</h1>
        </div>
        
        <div className="flex items-center space-x-4">
          <div className="text-sm text-gray-500">
            Last updated: {formatDateTime(dashboardData.lastUpdated)}
          </div>
          
          <div className="flex space-x-2">
            <button 
              onClick={() => setTimeRange('day')}
              className={`px-3 py-1 rounded ${timeRange === 'day' ? 'bg-blue-600 text-white' : 'bg-gray-200'}`}
            >
              Day
            </button>
            <button 
              onClick={() => setTimeRange('week')}
              className={`px-3 py-1 rounded ${timeRange === 'week' ? 'bg-blue-600 text-white' : 'bg-gray-200'}`}
            >
              Week
            </button>
            <button 
              onClick={() => setTimeRange('month')}
              className={`px-3 py-1 rounded ${timeRange === 'month' ? 'bg-blue-600 text-white' : 'bg-gray-200'}`}
            >
              Month
            </button>
          </div>
          
          <button onClick={() => fetchDashboardData()} className="p-2 rounded-full hover:bg-gray-100">
            <RefreshCw className="h-5 w-5 text-gray-600" />
          </button>
          
          <button onClick={() => setShowSettings(!showSettings)} className="p-2 rounded-full hover:bg-gray-100">
            <Sliders className="h-5 w-5 text-gray-600" />
          </button>
          
          <button onClick={exportData} className="p-2 rounded-full hover:bg-gray-100">
            <Download className="h-5 w-5 text-gray-600" />
          </button>
        </div>
      </header>
      
      {/* Settings panel (conditionally rendered) */}
      {showSettings && (
        <div className="bg-white p-4 rounded-lg shadow-md mb-6">
          <h2 className="text-lg font-semibold mb-3">Dashboard Settings</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Area Filter</label>
              <select 
                value={areaFilter}
                onChange={(e) => setAreaFilter(e.target.value)}
                className="w-full p-2 border border-gray-300 rounded-md"
              >
                <option value="all">All Areas</option>
                <option value="downtown">Downtown</option>
                <option value="north">North District</option>
                <option value="south">South District</option>
                <option value="east">East District</option>
                <option value="west">West District</option>
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Refresh Interval</label>
              <select 
                defaultValue={REFRESH_INTERVAL / 1000}
                onChange={(e) => {
                  const newInterval = parseInt(e.target.value) * 1000;
                  if (refreshTimerRef.current) {
                    clearInterval(refreshTimerRef.current);
                    refreshTimerRef.current = setInterval(fetchDashboardData, newInterval);
                  }
                }}
                className="w-full p-2 border border-gray-300 rounded-md"
              >
                <option value="30">30 seconds</option>
                <option value="60">1 minute</option>
                <option value="300">5 minutes</option>
                <option value="600">10 minutes</option>
              </select>
            </div>
          </div>
          
          <div className="mt-4">
            <button 
              onClick={() => setShowSettings(false)} 
              className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
            >
              Apply Settings
            </button>
          </div>
        </div>
      )}
      
      {/* Error alert */}
      {error && (
        <div className="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 mb-6 rounded-md shadow-md">
          <div className="flex items-center">
            <AlertTriangle className="h-5 w-5 mr-2" />
            <p>{error}</p>
          </div>
        </div>
      )}
      
      {/* KPI Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-4 gap-4 mb-6">
        {/* Current Congestion Index */}
        <div className="bg-white p-4 rounded-lg shadow-md">
          <div className="flex justify-between items-start">
            <div>
              <h3 className="text-sm font-medium text-gray-500">Current Congestion</h3>
              <div className="flex items-baseline">
                <span className="text-3xl font-semibold">
                  {dashboardData.congestionIndex}%
                </span>
                {getChangeIndicator(5)} {/* Example change value */}
              </div>
            </div>
            <div 
              className="w-10 h-10 rounded-full flex items-center justify-center"
              style={{ backgroundColor: getStatusColor(dashboardData.trafficStatus) }}
            >
              <Car className="h-6 w-6 text-white" />
            </div>
          </div>
          <div className="mt-2 text-sm font-medium" style={{ color: getStatusColor(dashboardData.trafficStatus) }}>
            {dashboardData.trafficStatus.charAt(0).toUpperCase() + dashboardData.trafficStatus.slice(1)} traffic
          </div>
        </div>
        
        {/* Active Incidents */}
        <div className="bg-white p-4 rounded-lg shadow-md">
          <div className="flex justify-between items-start">
            <div>
              <h3 className="text-sm font-medium text-gray-500">Active Incidents</h3>
              <div className="flex items-baseline">
                <span className="text-3xl font-semibold">
                  {dashboardData.incidentCount}
                </span>
              </div>
            </div>
            <div className="w-10 h-10 rounded-full bg-orange-100 flex items-center justify-center">
              <AlertTriangle className="h-6 w-6 text-orange-500" />
            </div>
          </div>
          <div className="mt-2 text-sm text-gray-600">
            {dashboardData.trafficIncidents && dashboardData.trafficIncidents.length > 0 
              ? `Latest: ${dashboardData.trafficIncidents[0].type} - ${dashboardData.trafficIncidents[0].location}`
              : 'No recent incidents'}
          </div>
        </div>
        
        {/* Congestion Reduction */}
        <div className="bg-white p-4 rounded-lg shadow-md">
          <div className="flex justify-between items-start">
            <div>
              <h3 className="text-sm font-medium text-gray-500">Congestion Reduction</h3>
              <div className="flex items-baseline">
                <span className="text-3xl font-semibold">
                  {dashboardData.optimizationStats?.congestionReduction || 0}%
                </span>
                <span className="text-sm text-green-500 ml-2">vs baseline</span>
              </div>
            </div>
            <div className="w-10 h-10 rounded-full bg-green-100 flex items-center justify-center">
              <Activity className="h-6 w-6 text-green-500" />
            </div>
          </div>
          <div className="mt-2 text-sm text-gray-600">
            AI optimization active on {dashboardData.optimizationStats?.trafficLightOptimization || 0}% of signals
          </div>
        </div>
        
        {/* Route Suggestions */}
        <div className="bg-white p-4 rounded-lg shadow-md">
          <div className="flex justify-between items-start">
            <div>
              <h3 className="text-sm font-medium text-gray-500">Route Suggestion Acceptance</h3>
              <div className="flex items-baseline">
                <span className="text-3xl font-semibold">
                  {dashboardData.optimizationStats?.routeSuggestionAcceptance || 0}%
                </span>
              </div>
            </div>
            <div className="w-10 h-10 rounded-full bg-blue-100 flex items-center justify-center">
              <MapPin className="h-6 w-6 text-blue-500" />
            </div>
          </div>
          <div className="mt-2 text-sm text-gray-600">
            {(dashboardData.optimizationStats?.fuelSavings || 0)}% estimated fuel savings
          </div>
        </div>
      </div>
      
      {/* Main Dashboard Content */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column */}
        <div className="lg:col-span-2 space-y-6">
          {/* Traffic Flow Chart */}
          <div className="bg-white p-4 rounded-lg shadow-md">
            <h2 className="text-lg font-semibold mb-4">Traffic Flow and Congestion</h2>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={dashboardData.trafficFlow}
                  margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" />
                  <YAxis yAxisId="left" />
                  <YAxis yAxisId="right" orientation="right" />
                  <Tooltip />
                  <Legend />
                  <Line
                    yAxisId="left"
                    type="monotone"
                    dataKey="flow"
                    name="Vehicles/Hour"
                    stroke="#2196f3"
                    activeDot={{ r: 8 }}
                  />
                  <Line
                    yAxisId="right"
                    type="monotone"
                    dataKey="congestion"
                    name="Congestion %"
                    stroke="#f44336"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
          
          {/* Congestion Hotspots */}
          <div className="bg-white p-4 rounded-lg shadow-md">
            <h2 className="text-lg font-semibold mb-4">Congestion Hotspots</h2>
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Location
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Severity
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Duration
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Status
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {dashboardData.congestionHotspots && dashboardData.congestionHotspots.map((hotspot, index) => (
                    <tr key={index}>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                        {hotspot.location}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        <div className="flex items-center">
                          <div className="w-full bg-gray-200 rounded-full h-2.5">
                            <div 
                              className="h-2.5 rounded-full" 
                              style={{ 
                                width: `${hotspot.severity}%`,
                                backgroundColor: hotspot.severity > 70 
                                  ? STATUS_COLORS.heavy 
                                  : hotspot.severity > 40 
                                    ? STATUS_COLORS.moderate 
                                    : STATUS_COLORS.low
                              }}
                            ></div>
                          </div>
                          <span className="ml-2">{hotspot.severity}%</span>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {hotspot.duration}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full 
                          ${hotspot.severity > 70 
                            ? 'bg-red-100 text-red-800' 
                            : hotspot.severity > 40 
                              ? 'bg-yellow-100 text-yellow-800' 
                              : 'bg-green-100 text-green-800'}`}
                        >
                          {hotspot.severity > 70 
                            ? 'Critical' 
                            : hotspot.severity > 40 
                              ? 'Significant' 
                              : 'Moderate'}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
          
          {/* Traffic Incidents */}
          <div className="bg-white p-4 rounded-lg shadow-md">
            <h2 className="text-lg font-semibold mb-4">Active Traffic Incidents</h2>
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Type
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Location
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Time
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Status
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Severity
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {dashboardData.trafficIncidents && dashboardData.trafficIncidents.map((incident, index) => (
                    <tr key={index}>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                        {incident.type}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {incident.location}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {incident.time}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full 
                          ${incident.status === 'Active' 
                            ? 'bg-red-100 text-red-800' 
                            : incident.status === 'Clearing' 
                              ? 'bg-yellow-100 text-yellow-800' 
                              : 'bg-green-100 text-green-800'}`}
                        >
                          {incident.status}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full 
                          ${incident.severity === 'Major' 
                            ? 'bg-red-100 text-red-800' 
                            : incident.severity === 'Moderate' 
                              ? 'bg-yellow-100 text-yellow-800' 
                              : 'bg-green-100 text-green-800'}`}
                        >
                          {incident.severity}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
        
        {/* Right Column */}
        <div className="space-y-6">
          {/* Transportation Mode Distribution */}
          <div className="bg-white p-4 rounded-lg shadow-md">
            <h2 className="text-lg font-semibold mb-4">Transportation Mode Distribution</h2>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={dashboardData.modeDistribution}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {dashboardData.modeDistribution && dashboardData.modeDistribution.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={CHART_COLORS[index % CHART_COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(value) => `${value}%`} />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>
          
          {/* AI Optimization Stats */}
          <div className="bg-white p-4 rounded-lg shadow-md">
            <h2 className="text-lg font-semibold mb-4">AI Optimization Impact</h2>
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={[
                    { name: 'Traffic Signals', value: dashboardData.optimizationStats?.trafficLightOptimization || 0 },
                    { name: 'Route Acceptance', value: dashboardData.optimizationStats?.routeSuggestionAcceptance || 0 },
                    { name: 'Congestion↓', value: dashboardData.optimizationStats?.congestionReduction || 0 },
                    { name: 'Fuel Savings', value: dashboardData.optimizationStats?.fuelSavings || 0 },
                    { name: 'Emissions↓', value: dashboardData.optimizationStats?.emissionReduction || 0 }
                  ]}
                  margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis label={{ value: 'Percentage %', angle: -90, position: 'insideLeft' }} />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="value" name="Improvement %" fill="#2196f3" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
          
          {/* Weekly Congestion Trend */}
          <div className="bg-white p-4 rounded-lg shadow-md">
            <h2 className="text-lg font-semibold mb-4">Weekly Congestion Trend</h2>
            <div className="h-60">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart
                  data={[
                    { day: 'Mon', peak: 63, average: 42 },
                    { day: 'Tue', peak: 58, average: 40 },
                    { day: 'Wed', peak: 65, average: 43 },
                    { day: 'Thu', peak: 74, average: 48 },
                    { day: 'Fri', peak: 81, average: 52 },
                    { day: 'Sat', peak: 48, average: 30 },
                    { day: 'Sun', peak: 42, average: 27 }
                  ]}
                  margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="day" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Area type="monotone" dataKey="peak" name="Peak Congestion %" stroke="#f44336" fill="#ffcdd2" />
                  <Area type="monotone" dataKey="average" name="Average Congestion %" stroke="#2196f3" fill="#bbdefb" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>
      
      {/* Footer */}
      <footer className="mt-8 text-center text-gray-500 text-sm">
        <p>Smart City Traffic Management Dashboard</p>
        <p>Data refreshes every minute. Last update: {formatDateTime(dashboardData.lastUpdated)}</p>
      </footer>
    </div>
  );
};

export default TrafficDashboard;

/**
 * TODOs and Future Enhancements:
 * 
 * 1. Implement real-time WebSocket connection for live updates
 * 2. Add interactive city map with traffic overlay visualization
 * 3. Create drill-down views for specific intersections or corridors
 * 4. Implement traffic camera feeds integration
 * 5. Add predictive analytics section showing forecasted congestion
 * 6. Create mobile-responsive version for field operators
 * 7. Add user authentication for admin features
 * 8. Implement dark mode toggle for night operations
 * 9. Create anomaly detection alerts for unusual traffic patterns
 * 10. Add comparative views (current vs. historical patterns)
 * 11. Create API documentation for third-party integrations
 * 12. Add multi-language support for international deployments
 * 13. Implement dashboard customization options (arrangement, visible widgets)
 */
