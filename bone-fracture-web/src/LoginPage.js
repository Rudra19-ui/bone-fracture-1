import React, { useState } from 'react';
import { motion } from 'framer-motion';
import './LoginPage.css';

const LoginPage = ({ onLogin }) => {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    password: '',
    userType: 'doctor'
  });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');

    // Simulate login process
    setTimeout(() => {
      if (formData.name && formData.email && formData.password) {
        onLogin({
          name: formData.name,
          email: formData.email,
          userType: formData.userType
        });
      } else {
        setError('Please fill in all fields');
        setIsLoading(false);
      }
    }, 1500);
  };

  return (
    <div className="login-page">
      <div className="login-background">
        <div className="floating-shapes">
          {[...Array(15)].map((_, i) => (
            <div key={i} className={`shape shape-${i % 4}`}></div>
          ))}
        </div>
      </div>

      <div className="login-container">
        <motion.div
          className="login-card"
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          <div className="login-header">
            <div className="logo-section">
              <div className="logo-icon">🦴</div>
              <h1>Deep learning classification of fracture bones using ViT</h1>
              <p>Advanced Medical AI Platform</p>
            </div>
          </div>

          <form onSubmit={handleSubmit} className="login-form">
            <div className="form-group">
              <label htmlFor="name">Full Name</label>
              <input
                type="text"
                id="name"
                name="name"
                value={formData.name}
                onChange={handleInputChange}
                placeholder="Enter your full name"
                required
              />
            </div>

            <div className="form-group">
              <label htmlFor="email">Email Address</label>
              <input
                type="email"
                id="email"
                name="email"
                value={formData.email}
                onChange={handleInputChange}
                placeholder="Enter your email"
                required
              />
            </div>

            <div className="form-group">
              <label htmlFor="password">Password</label>
              <input
                type="password"
                id="password"
                name="password"
                value={formData.password}
                onChange={handleInputChange}
                placeholder="Enter your password"
                required
              />
            </div>

            <div className="form-group">
              <label htmlFor="userType">User Type</label>
              <select
                id="userType"
                name="userType"
                value={formData.userType}
                onChange={handleInputChange}
              >
                <option value="doctor">Doctor</option>
                <option value="radiologist">Radiologist</option>
                <option value="patient">Patient</option>
                <option value="admin">Administrator</option>
              </select>
            </div>

            {error && (
              <motion.div
                className="error-message"
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
              >
                {error}
              </motion.div>
            )}

            <button
              type="submit"
              className="login-btn"
              disabled={isLoading}
            >
              {isLoading ? (
                <div className="loading-spinner">
                  <div className="spinner"></div>
                  Signing in...
                </div>
              ) : (
                'Sign In'
              )}
            </button>
          </form>

          <div className="login-footer">
            <div className="demo-credentials">
              <h4>Demo Credentials:</h4>
              <p><strong>Doctor:</strong> Dr. Smith / doctor@vitfracturemodel.com / password123</p>
              <p><strong>Radiologist:</strong> Dr. Johnson / radiologist@vitfracturemodel.com / password123</p>
              <p><strong>Patient:</strong> John Doe / patient@vitfracturemodel.com / password123</p>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default LoginPage;
