import React, { useEffect, useState, useCallback, useRef } from "react";
import {
  createTheme,
  ThemeProvider,
  CssBaseline,
  AppBar,
  Toolbar,
  Container,
  Box,
  IconButton,
  Typography,
  Paper,
  Button,
  TextField,
  CircularProgress,
  Divider,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Modal,
  Tabs,
  Tab,
} from "@mui/material";
import LightModeIcon from "@mui/icons-material/LightMode";
import DarkModeIcon from "@mui/icons-material/DarkMode";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip as ChartTooltip,
  Legend as ChartLegend,
  TimeScale,
} from "chart.js";
import zoomPlugin from "chartjs-plugin-zoom";
import { Line } from "react-chartjs-2";
import { LazyLog } from "react-lazylog";
import { io } from 'socket.io-client';

// ============== REGISTER CHART.JS + PLUGIN ==============
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  ChartTooltip,
  ChartLegend,
  TimeScale,
  zoomPlugin
);

/**
 * Custom hook: fetch /scalars + /training_state, handle config logic,
 * store chart zoom in localStorage
 */
function useDashboardData() {
  const [scalars, setScalars] = useState({});
  const [trainingState, setTrainingState] = useState(null);
  const [currentConfig, setCurrentConfig] = useState(null);
  const [inputConfig, setInputConfig] = useState({
    batch_size: "",
    learning_rate: "",
    running: false
  });
  const [isChanging, setIsChanging] = useState(false);
  const [error, setError] = useState(null);
  const [socket, setSocket] = useState(null);
  const [scalarData, setScalarData] = useState({});
  const [inputInitialized, setInputInitialized] = useState(false);
  const [connectionState, setConnectionState] = useState({
    isConnected: false,
    error: null,
    reconnectAttempt: 0
  });

  // Zoom/pan ranges stored in localStorage
  const [zoomRanges, setZoomRanges] = useState({});

  // Initialize input values ONLY ONCE when current config is first loaded
  useEffect(() => {
    if (currentConfig && !inputInitialized) {
      setInputConfig({
        batch_size: String(currentConfig.batch_size),
        learning_rate: String(currentConfig.learning_rate),
        running: currentConfig.running
      });
      setInputInitialized(true);
    }
  }, [currentConfig, inputInitialized]);

  // Initialize WebSocket connection and fetch initial data
  useEffect(() => {
    let reconnectTimer;

    const connectSocket = () => {
      const newSocket = io('http://localhost:5100', {
        transports: ['websocket'],
        reconnection: false,  // We'll handle reconnection manually
      });

      newSocket.on('connect', () => {
        console.log('Connected to server');
        setConnectionState(prev => ({
          ...prev,
          isConnected: true,
          error: null,
          reconnectAttempt: 0
        }));
        newSocket.emit('request_state');
      });

      newSocket.on('connect_error', (error) => {
        console.log('Connection Error:', error);
        setConnectionState(prev => ({
          ...prev,
          isConnected: false,
          error: 'Failed to connect to server',
          reconnectAttempt: prev.reconnectAttempt + 1
        }));
      });

      newSocket.on('disconnect', (reason) => {
        console.log('Disconnected:', reason);
        setConnectionState(prev => ({
          ...prev,
          isConnected: false,
          error: `Disconnected: ${reason}`,
          reconnectAttempt: prev.reconnectAttempt + 1
        }));
      });

      newSocket.on('state_update', (data) => {
        console.log('Received state update:', data);
        setTrainingState(data.state);
        setCurrentConfig(data.config);
      });

      newSocket.on('config_updated', (response) => {
        console.log('Config update response:', response);
        if (response.status === 'error') {
          setError(response.message);
        }
        setIsChanging(false);
      });

      newSocket.on('state_error', (error) => {
        setError(error.message);
      });

      newSocket.on('scalar_update', (newData) => {
        setScalarData(newData);
      });

      setSocket(newSocket);
    };

    const startReconnectTimer = () => {
      reconnectTimer = setInterval(() => {
        if (!connectionState.isConnected) {
          console.log('Attempting to reconnect...');
          connectSocket();
        }
      }, 5000);
    };

    connectSocket();
    startReconnectTimer();

    return () => {
      clearInterval(reconnectTimer);
      if (socket) socket.disconnect();
    };
  }, [connectionState.isConnected]);

  // Fetch scalars separately
  useEffect(() => {
    fetchScalars();
  }, []);

  // == Zoom from localStorage ==
  useEffect(() => {
    const saved = localStorage.getItem("chartZoomRanges");
    if (saved) {
      try {
        setZoomRanges(JSON.parse(saved));
      } catch (err) {
        console.error("Failed to parse chartZoomRanges:", err);
      }
    }
  }, []);
  useEffect(() => {
    localStorage.setItem("chartZoomRanges", JSON.stringify(zoomRanges));
  }, [zoomRanges]);

  function fetchScalars() {
    fetch("http://localhost:5100/scalars")
      .then((r) => r.json())
      .then((data) => setScalars(data))
      .catch((err) => setError(err.message));
  }

  const handleInputChange = (key, value) => {
    if (key === 'learning_rate' && !trainingState?.learning_rate) {
      console.log('Cannot modify learning rate before training starts');
      return;
    }

    setInputConfig(prev => ({
      ...prev,
      [key]: value
    }));
    setIsChanging(true);
  };

  const applyChanges = () => {
    if (!socket) return;
    
    const newConfig = {
      ...currentConfig,
      batch_size: parseInt(inputConfig.batch_size),
      learning_rate: parseFloat(inputConfig.learning_rate),
      running: inputConfig.running
    };
    
    socket.emit('update_config', newConfig);
  };

  const handleStop = () => {
    if (!socket) return;
    setInputConfig(prev => ({ ...prev, running: false }));
    socket.emit('update_config', { ...currentConfig, running: false });
  };

  const handleStart = () => {
    if (!socket) return;
    setInputConfig(prev => ({ ...prev, running: true }));
    socket.emit('update_config', { ...currentConfig, running: true });
  };

  // == Zoom/Pan events ==
  function handleZoomPanComplete(metricKey, chart) {
    const xScale = chart.scales.x;
    const yScale = chart.scales.y;
    setZoomRanges((prev) => ({
      ...prev,
      [metricKey]: {
        xMin: xScale.min,
        xMax: xScale.max,
        yMin: yScale.min,
        yMax: yScale.max,
      },
    }));
  }
  function getChartOptions(metricKey) {
    const st = zoomRanges[metricKey];
    let base = {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: "top" },
        title: { display: false },
        zoom: {
          zoom: {
            wheel: { enabled: true }, // mouse wheel
            pinch: { enabled: true }, // pinch gesture
            mode: "xy",
          },
          pan: {
            enabled: true,
            mode: "xy",
          },
          onZoomComplete: ({ chart }) => handleZoomPanComplete(metricKey, chart),
          onPanComplete: ({ chart }) => handleZoomPanComplete(metricKey, chart),
        },
      },
      scales: {
        x: { type: "linear", title: { display: true, text: "Step" } },
        y: { title: { display: true, text: "Value" } },
      },
    };
    if (st) {
      base.scales.x.min = st.xMin;
      base.scales.x.max = st.xMax;
      base.scales.y.min = st.yMin;
      base.scales.y.max = st.yMax;
    }
    return base;
  }

  // Helper to check if changes are pending
  const isPending = useCallback(() => {
    if (!trainingState || !currentConfig) return false;
    
    // Compare state with config to see if simulator has caught up
    return trainingState.running !== currentConfig.running ||
           String(trainingState.batch_size) !== String(currentConfig.batch_size) ||
           (String(trainingState.learning_rate) !== String(currentConfig.learning_rate) && currentConfig.learning_rate &&String(currentConfig.learning_rate)!="" && trainingState.running);
  }, [trainingState, currentConfig]);

  // Update chart data when scalarData changes
  useEffect(() => {
    if (Object.keys(scalarData).length > 0) {
      // Update your charts here using scalarData
      // Example for loss chart:
      if (scalarData.training_loss) {
        setScalars(prev => ({
          ...prev,
          training_loss: scalarData.training_loss
        }));
      }
    }
  }, [scalarData]);

  return {
    scalars,
    trainingState,
    currentConfig,
    inputConfig,
    isChanging,
    error,
    connectionState,
    isPending,
    handleInputChange,
    applyChanges,
    getChartOptions,
    handleStop,
    handleStart,
  };
}

// Add this helper function to group metrics by prefix
function groupMetricsByPrefix(scalars) {
  const groups = {
    performance: {},
    raw: {},
    tpr: {},
    fairness: {},
    bias: {},
    others: {}  // For metrics that don't match any known prefix
  };

  Object.entries(scalars).forEach(([key, value]) => {
    const parts = key.split('/');
    const prefix = parts[0];
    
    if (groups.hasOwnProperty(prefix)) {
      groups[prefix][key] = value;
    } else {
      groups.others[key] = value;
    }
  });

  return groups;
}

// Create a new MetricsTabs component
function MetricsTabs({ scalars, getChartOptions }) {
  const [currentTab, setCurrentTab] = useState(0);
  const groups = Object.keys(scalars);
  
  const handleTabChange = (event, newValue) => {
    setCurrentTab(newValue);
  };

  const renderMetricCharts = (metrics) => {
    return Object.entries(metrics).map(([metricKey, arr]) => {
      if (!Array.isArray(arr) || arr.length === 0) {
        return (
          <Box key={metricKey} sx={{ p: 2 }}>
            <Typography>{metricKey}</Typography>
            <Typography>No data</Typography>
          </Box>
        );
      }

      const chartData = {
        labels: arr.map((pt) => pt.step),
        datasets: [{
          label: metricKey,
          data: arr.map((pt) => pt.value),
          borderColor: "#8884d8",
          backgroundColor: "#8884d8",
        }],
      };

      return (
        <Box key={metricKey} sx={{ mb: 3 }}>
          <Typography fontWeight="bold" sx={{ mb: 1 }}>
            {metricKey}
          </Typography>
          <Box sx={{ width: "100%", height: 300 }}>
            <Line data={chartData} options={getChartOptions(metricKey)} />
          </Box>
        </Box>
      );
    });
  };

  return (
    <Box sx={{ width: '100%' }}>
      <Tabs 
        value={currentTab} 
        onChange={handleTabChange}
        variant="scrollable"
        scrollButtons="auto"
        sx={{ borderBottom: 1, borderColor: 'divider', mb: 2 }}
      >
        {groups.map((group, index) => (
          <Tab key={group} label={group.charAt(0).toUpperCase() + group.slice(1)} />
        ))}
      </Tabs>

      {groups.map((group, index) => (
        <TabPanel key={group} value={currentTab} index={index}>
          {renderMetricCharts(scalars[group])}
        </TabPanel>
      ))}
    </Box>
  );
}

// Helper component for tabs
function TabPanel(props) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`metrics-tabpanel-${index}`}
      aria-labelledby={`metrics-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

/**
 * 5) Single-Page layout. 
 *    - Top bar for brand & dark mode toggle
 *    - Fluid container (maxWidth="xl") for adaptive full-screen
 *    - 3 Paper sections stacked vertically: metrics, configs, logs
 */
export default function MainApp() {
  const [darkMode, setDarkMode] = useState(true);
  const toggleDarkMode = () => setDarkMode((prev) => !prev);

  const theme = createTheme({
    palette: {
      mode: darkMode ? "dark" : "light",
    },
    typography: {
      fontFamily: `"Inter", "Roboto", "Helvetica", "Arial", sans-serif`,
      fontSize: 14,
    },
  });

  // Data from the custom hook
  const {
    scalars,
    trainingState,
    currentConfig,
    inputConfig,
    isChanging,
    error,
    connectionState,
    isPending,
    handleInputChange,
    applyChanges,
    getChartOptions,
    handleStop,
    handleStart,
  } = useDashboardData();

  const renderStateValue = (value) => {
    if (value === null || value === undefined) return 'N/A';
    if (typeof value === 'boolean') return value.toString();
    if (typeof value === 'number') return value.toFixed(6);
    if (typeof value === 'string') return value;
    if (typeof value === 'object') return JSON.stringify(value);
    return value.toString();
  };

  const logsContainerRef = useRef(null);

  const scrollToBottom = () => {
    if (logsContainerRef.current) {
      logsContainerRef.current.scrollTop = logsContainerRef.current.scrollHeight;
    }
  };

  // Scroll to bottom when logs update
  useEffect(() => {
    if (trainingState?.stdout) {
      scrollToBottom();
    }
  }, [trainingState?.stdout]);

  // If data not loaded => spinner
  if (!trainingState) {
    return (
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <Typography sx={{ mt: 8, ml: 2 }}>Loading...</Typography>
      </ThemeProvider>
    );
  }

  // Overlay for Start/Stop
  const stateRunning = trainingState.running;
  const configRunning = inputConfig.running;
  let overlayText = "";
  if (isChanging) {
    if (configRunning && !stateRunning) overlayText = "Starting up…";
    if (!configRunning && stateRunning) overlayText = "Shutting down…";
  }
  const trainingStopped = !stateRunning && !configRunning && !isChanging;

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />

      {/* Top Bar */}
      <AppBar position="fixed">
        <Toolbar>
          <Typography variant="h5" sx={{ flexGrow: 1, fontWeight: "bold" }}>
            Fairness Training Dashboard
          </Typography>
          <IconButton onClick={toggleDarkMode} sx={{ color: "inherit" }}>
            {darkMode ? <LightModeIcon /> : <DarkModeIcon />}
          </IconButton>
        </Toolbar>
      </AppBar>

      {/* Container => fluid & responsive up to 'xl' screens */}
      <Container 
        maxWidth={false}
        sx={{ 
          mt: 12, 
          minHeight: 'calc(100vh - 64px - 32px)', 
          position: "relative", 
          pb: 4,
          px: { xs: 3, sm: 4, md: 6 }
        }}
      >
        {/* Show errors */}
        {error && (
          <Typography color="error" sx={{ mb: 2 }}>
            {error}
          </Typography>
        )}

        {/* Pending Changes Indicator */}
        {isPending() && (
          <Paper 
            sx={{ 
              p: 2, 
              mb: 2, 
              backgroundColor: 'warning.main',
              color: 'warning.contrastText',
              display: 'flex',
              alignItems: 'center',
              gap: 2
            }}
          >
            <CircularProgress 
              size={20} 
              thickness={5} 
              sx={{ color: 'warning.contrastText' }} 
            />
            <Typography>
              Changes pending... Waiting for trainer to update
            </Typography>
          </Paper>
        )}

        {/* Start/Stop overlay */}
        {overlayText && (
          <Box
            sx={{
              position: "absolute",
              top: 0,
              left: 0,
              width: "100%",
              height: "100%",
              bgColor: "rgba(0,0,0,0.3)",
              zIndex: 9999,
              color: "white",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: 32,
              fontWeight: "bold",
            }}
          >
            {overlayText}
          </Box>
        )}

        {/* Buffer space */}
        <Box sx={{ mb: 4 }} />

        <Box sx={{ 
          display: "flex", 
          gap: 2, 
          minHeight: '100%',
          pb: 4,
          mx: 'auto',
          maxWidth: '1800px',
        }}>
          {/* Left side: Logs and Metrics */}
          <Box sx={{ 
            flex: '1 1 auto', 
            overflow: 'auto',
            minWidth: '0',
            maxWidth: '70%',  // Increased to 70%
            width: '70%',     // Set preferred width
            mr: 2
          }}>
            {/* Training Logs - Now First */}
            <Paper sx={{ p: 2, mb: 2 }} elevation={2}>
              <Typography variant="h6" fontWeight="bold" gutterBottom>
                Training Logs
              </Typography>
              <Box
                ref={logsContainerRef}
                sx={{
                  backgroundColor: 'black',
                  color: 'lightgreen',
                  p: 2,
                  borderRadius: 1,
                  fontFamily: 'monospace',
                  maxHeight: '400px',
                  overflowY: 'auto',
                  overflowX: 'auto',
                  whiteSpace: 'pre-wrap',
                  width: '100%',
                  minWidth: '500px',
                }}
              >
                {trainingState?.stdout || 'No logs yet.'}
              </Box>
            </Paper>

            {/* Metrics Paper */}
            <Paper sx={{ p: 2 }} elevation={2}>
              <Typography variant="h6" fontWeight="bold" gutterBottom>
                Metrics
              </Typography>
              {Object.keys(scalars).length === 0 ? (
                <Box sx={{ p: 2 }}>
                  <CircularProgress />
                </Box>
              ) : (
                <MetricsTabs scalars={scalars} getChartOptions={getChartOptions} />
              )}
            </Paper>
          </Box>

          {/* Right side: Config Panel */}
          <Box sx={{ 
            width: { xs: '350px', md: '400px', lg: '450px' },  // Responsive width
            flexShrink: 0,
            position: 'sticky',
            alignSelf: 'flex-start',
            ml: 2,
            mr: { xs: 2, sm: 3, md: 4 }
          }}>
            <Paper 
              sx={{ 
                p: 2,
                height: 'auto',
                position: 'relative',
                display: 'flex',
                flexDirection: 'column'
              }} 
              elevation={2}
            >
              <Box sx={{ position: 'relative', zIndex: 0 }}>
                <Typography variant="h6" fontWeight="bold" gutterBottom>
                  Configs
                </Typography>
                
                {/* Current Config as table */}
                <TableContainer sx={{ maxHeight: 'none' }}>  {/* Remove max height */}
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell colSpan={2} align="center">
                          Current Settings
                        </TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {trainingState && Object.entries(trainingState)
                        .filter(([key]) => key !== 'stdout')
                        .sort(([keyA], [keyB]) => keyA.localeCompare(keyB))
                        .map(([key, value]) => (
                          <TableRow key={key}>
                            <TableCell 
                              sx={{ 
                                textTransform: 'capitalize',
                                fontWeight: 'medium'
                              }}
                            >
                              {key.replace(/_/g, ' ')}
                            </TableCell>
                            <TableCell>{renderStateValue(value)}</TableCell>
                          </TableRow>
                        ))}
                    </TableBody>
                  </Table>
                </TableContainer>

                <Divider sx={{ my: 2 }} />

                {/* Config Controls - disabled when pending */}
                <Box sx={{ 
                  opacity: isPending() ? 0.5 : 1,
                  pointerEvents: isPending() ? 'none' : 'auto'
                }}>
                  <Box sx={{ mb: 1 }}>
                    Running: {String(inputConfig.running)}
                  </Box>
                  {trainingState?.running ? (
                    <Button
                      variant="outlined"
                      sx={{ mb: 2 }}
                      onClick={handleStop}
                      disabled={isPending()}
                    >
                      Stop
                    </Button>
                  ) : (
                    <Button
                      variant="outlined"
                      sx={{ mb: 2 }}
                      onClick={handleStart}
                      disabled={isPending()}
                    >
                      Start
                    </Button>
                  )}
                  <TextField
                    label="Batch Size"
                    type="number"
                    fullWidth
                    margin="dense"
                    value={inputConfig.batch_size}
                    onChange={(e) => handleInputChange("batch_size", e.target.value)}
                    disabled={isPending()}
                    sx={{ mb: 1 }}
                  />
                  <TextField
                    label="Learning Rate"
                    type="number"
                    fullWidth
                    margin="dense"
                    value={inputConfig.learning_rate}
                    onChange={(e) => handleInputChange("learning_rate", e.target.value)}
                    disabled={isPending()}
                  />
                  <Button 
                    variant="contained" 
                    sx={{ mt: 2 }} 
                    onClick={applyChanges}
                    disabled={isPending()}
                  >
                    Apply
                  </Button>
                </Box>
              </Box>

              {/* Pending Changes Overlay */}
              {isPending() && (
                <Box
                  sx={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    right: 0,
                    bottom: 0,
                    backgroundColor: 'rgba(0, 0, 0, 0.7)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    zIndex: 1,
                    backdropFilter: 'blur(2px)',
                  }}
                >
                  <Box sx={{ textAlign: 'center' }}>
                    <CircularProgress sx={{ mb: 2 }} />
                    <Typography
                      variant="h6"
                      sx={{
                        color: 'white',
                        fontWeight: 'bold',
                        animation: 'bounce 1s infinite',
                        '@keyframes bounce': {
                          '0%, 100%': { transform: 'translateY(0)' },
                          '50%': { transform: 'translateY(-10px)' },
                        },
                      }}
                    >
                      Changes Pending...
                    </Typography>
                  </Box>
                </Box>
              )}

              {/* Training Stopped Overlay - only show when not pending */}
              {trainingStopped && !isPending() && (
                <Box
                  sx={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    right: 0,
                    bottom: 0,
                    backgroundColor: 'rgba(0, 0, 0, 0.7)',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: 3,
                    zIndex: 1,
                    backdropFilter: 'blur(2px)',
                  }}
                >
                  <Typography
                    variant="h5"
                    sx={{
                      color: 'white',
                      fontWeight: 'bold',
                      textAlign: 'center',
                      animation: 'bounce 1s infinite',
                      '@keyframes bounce': {
                        '0%, 100%': { transform: 'translateY(0)' },
                        '50%': { transform: 'translateY(-10px)' },
                      },
                    }}
                  >
                    Training Stopped
                  </Typography>
                  <Button
                    variant="contained"
                    size="large"
                    onClick={handleStart}
                    sx={{
                      fontSize: '1.2rem',
                      px: 4,
                      py: 1.5,
                      backgroundColor: 'primary.main',
                      '&:hover': {
                        backgroundColor: 'primary.dark',
                      },
                      position: 'relative',
                    }}
                  >
                    Start Training
                  </Button>
                </Box>
              )}
            </Paper>
          </Box>
        </Box>
      </Container>

      {/* Single Connection/Loading Modal */}
      <Modal
        open={!connectionState.isConnected || !trainingState}
        aria-labelledby="connection-modal-title"
        aria-describedby="connection-modal-description"
      >
        <Box sx={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          width: 400,
          bgcolor: 'background.paper',
          borderRadius: 2,
          boxShadow: 24,
          p: 4,
        }}>
          <Typography id="connection-modal-title" variant="h6" component="h2" gutterBottom>
            Connection Error
          </Typography>
          <Typography id="connection-modal-description" sx={{ mt: 2, mb: 3 }}>
            {connectionState.error || 'Connecting to server...'}
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <CircularProgress size={20} />
            <Typography variant="body2" color="text.secondary">
              Reconnect attempt: {connectionState.reconnectAttempt}
            </Typography>
          </Box>
        </Box>
      </Modal>
    </ThemeProvider>
  );
}