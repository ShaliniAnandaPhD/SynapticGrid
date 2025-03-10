import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import SmartGridDashboard from './components/SmartGridDashboard';
import './App.css';

function App() {
  return (
    <Router>
      <div className="App">
        <Routes>
          <Route path="/" element={<SmartGridDashboard />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
