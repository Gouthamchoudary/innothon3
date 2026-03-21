import sys

content = open('dashboard2.py', 'r', encoding='utf-8').read()

# 1. Add import chatbot
content = content.replace(
    'from sklearn.ensemble import IsolationForest', 
    'from sklearn.ensemble import IsolationForest\nimport chatbot'
)

# 2. Add calendar enrichment to anomalies
enrichment = '''    df_clean, anomalies = run_anomaly_detection(df_m)
    if 'timestamp' in anomalies.columns:
        anomalies['calendar_event'] = anomalies['timestamp'].dt.strftime('%Y-%m-%d').apply(chatbot.get_calendar_event)'''
content = content.replace('    df_clean, anomalies = run_anomaly_detection(df_m)', enrichment)

# 3. Add hover text to anomaly scatter
scatter_repl = '''        fig_anom.add_trace(go.Scatter(
            x=anomalies['timestamp'], y=anomalies['active_power_kw'],
            mode='markers', name='Anomaly',
            text=anomalies.get('calendar_event', ''),
            hovertemplate="<b>%{x}</b><br>Power: %{y:.2f} kW<br>Event: %{text}<extra></extra>",
            marker=dict(color=ACCENT_DANGER, size=5, symbol='diamond',
                        line=dict(width=0.5, color='rgba(239,68,68,0.5)')),
        ))'''
content = content.replace('''        fig_anom.add_trace(go.Scatter(
            x=anomalies['timestamp'], y=anomalies['active_power_kw'],
            mode='markers', name='Anomaly',
            marker=dict(color=ACCENT_DANGER, size=5, symbol='diamond',
                        line=dict(width=0.5, color='rgba(239,68,68,0.5)')),
        ))''', scatter_repl)
        
# 4. Add display column for calendar_event
display_cols = "display_cols = ['timestamp', 'active_power_kw', 'current_avg', 'voltage_ll_avg',\n                    'power_factor', 'anomaly_score', 'severity', 'calendar_event']"
content = content.replace("display_cols = ['timestamp', 'active_power_kw', 'current_avg', 'voltage_ll_avg',\n                    'power_factor', 'anomaly_score', 'severity']", display_cols)

# 5. Populate col_chat
col_chat_code = '''

with col_chat:
    st.markdown('<p class="section-label" style="text-align:center;">🤖 AI Assistant</p>', unsafe_allow_html=True)
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "model", 
            "content": "Hi! I analyze the energy data and academic calendar. How can I help?"
        }]

    chat_container = st.container(height=650)
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    prompt = st.chat_input("Ask about the energy data...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
        
        # Build context safely
        anom_count = len(anomalies) if 'anomalies' in locals() else 'Unknown'
        pow_kw = avg_power if 'avg_power' in locals() else 0
        total_kwh = total_energy if 'total_energy' in locals() else 0
        context = f"Total Energy: {total_kwh:,.0f} kWh\\nAvg Power: {pow_kw:.2f} kW\\nAnomalies Detected: {anom_count}"
        
        with chat_container:
            with st.chat_message("model"):
                with st.spinner("Analyzing..."):
                    import chatbot
                    response = chatbot.get_ai_response(prompt, context, st.session_state.messages[:-1])
                    st.markdown(response)
        st.session_state.messages.append({"role": "model", "content": response})
'''
content += col_chat_code

with open('dashboard.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('Done!')
