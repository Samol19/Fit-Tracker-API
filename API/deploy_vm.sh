#!/bin/bash

# Script de despliegue rápido para VM en Google Cloud
# Uso: ./deploy_vm.sh

set -e

echo "🚀 Desplegando Exercise Pose API en VM..."

# Variables
ZONE="us-central1-a"
VM_NAME="exercise-pose-vm"
MACHINE_TYPE="e2-medium"  # Cambiar a c2-standard-4 para máxima velocidad

# Crear VM
echo "📦 Creando VM..."
gcloud compute instances create $VM_NAME \
  --zone=$ZONE \
  --machine-type=$MACHINE_TYPE \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=20GB \
  --boot-disk-type=pd-balanced \
  --tags=http-server,https-server

echo "⏳ Esperando que la VM inicie (30 segundos)..."
sleep 30

# Configurar VM
echo "⚙️ Configurando VM..."
gcloud compute ssh $VM_NAME --zone=$ZONE --command="
    set -e
    
    echo '📦 Instalando dependencias...'
    sudo apt-get update -qq
    sudo apt-get install -y python3-pip git nginx > /dev/null
    
    echo '📥 Clonando repositorio...'
    cd ~
    git clone https://github.com/PSLeon24/AI_Exercise_Pose_Feedback.git
    cd AI_Exercise_Pose_Feedback/API
    
    echo '🐍 Instalando paquetes Python...'
    pip3 install -q fastapi uvicorn[standard] joblib scikit-learn numpy scipy websockets
    
    echo '🔧 Configurando servicio systemd...'
    sudo bash -c 'cat > /etc/systemd/system/exercise-api.service << EOF
[Unit]
Description=Exercise Pose API
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/home/$USER/AI_Exercise_Pose_Feedback/API
ExecStart=/usr/local/bin/uvicorn main:app --host 0.0.0.0 --port 8080 --workers 2 --loop uvloop
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF'
    
    echo '🌐 Configurando Nginx...'
    sudo bash -c 'cat > /etc/nginx/sites-available/exercise-api << EOF
upstream exercise_api {
    server 127.0.0.1:8080;
}

server {
    listen 80;
    server_name _;
    
    location / {
        proxy_pass http://exercise_api;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \\\$http_upgrade;
        proxy_set_header Connection \"upgrade\";
        proxy_set_header Host \\\$host;
        proxy_set_header X-Real-IP \\\$remote_addr;
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
    }
}
EOF'
    
    sudo ln -sf /etc/nginx/sites-available/exercise-api /etc/nginx/sites-enabled/
    sudo rm -f /etc/nginx/sites-enabled/default
    sudo nginx -t
    sudo systemctl restart nginx
    
    echo '🚀 Iniciando servicio...'
    sudo systemctl daemon-reload
    sudo systemctl enable exercise-api
    sudo systemctl start exercise-api
    
    echo '✅ Configuración completa!'
"

# Configurar firewall
echo "🔥 Configurando firewall..."
gcloud compute firewall-rules create allow-http-80 \
  --allow tcp:80 \
  --target-tags http-server \
  --source-ranges 0.0.0.0/0 2>/dev/null || echo "Firewall ya existe"

# Obtener IP
echo ""
echo "============================================"
echo "✅ ¡Despliegue completado!"
echo "============================================"
IP=$(gcloud compute instances describe $VM_NAME --zone=$ZONE --format='get(networkInterfaces[0].accessConfigs[0].natIP)')
echo ""
echo "📍 Tu API está disponible en:"
echo "   http://$IP/"
echo ""
echo "🧪 Probar endpoints:"
echo "   curl http://$IP/"
echo "   curl http://$IP/models"
echo ""
echo "🔌 WebSocket endpoints:"
echo "   ws://$IP/ws/pushup"
echo "   ws://$IP/ws/squat"
echo "   ws://$IP/ws/plank"
echo ""
echo "💡 Comandos útiles:"
echo "   # Conectarse por SSH"
echo "   gcloud compute ssh $VM_NAME --zone=$ZONE"
echo ""
echo "   # Ver logs"
echo "   gcloud compute ssh $VM_NAME --zone=$ZONE --command='sudo journalctl -u exercise-api -f'"
echo ""
echo "   # Apagar VM (para ahorrar dinero)"
echo "   gcloud compute instances stop $VM_NAME --zone=$ZONE"
echo ""
echo "   # Encender VM"
echo "   gcloud compute instances start $VM_NAME --zone=$ZONE"
echo "============================================"
