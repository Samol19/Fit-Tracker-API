# Script para controlar la VM desde PowerShell
# Uso: .\vm_control.ps1 [start|stop|status|ssh|logs|ip]

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('start','stop','status','ssh','logs','ip','delete')]
    [string]$Action
)

$VM_NAME = "exercise-pose-vm"
$ZONE = "us-central1-a"

switch ($Action) {
    'start' {
        Write-Host "🟢 Encendiendo VM..." -ForegroundColor Green
        gcloud compute instances start $VM_NAME --zone=$ZONE
        Write-Host "✅ VM encendida. Esperando 30 segundos..." -ForegroundColor Green
        Start-Sleep -Seconds 30
        $IP = gcloud compute instances describe $VM_NAME --zone=$ZONE --format='get(networkInterfaces[0].accessConfigs[0].natIP)'
        Write-Host "📍 API disponible en: http://$IP/" -ForegroundColor Cyan
    }
    
    'stop' {
        Write-Host "🔴 Apagando VM..." -ForegroundColor Yellow
        gcloud compute instances stop $VM_NAME --zone=$ZONE
        Write-Host "✅ VM apagada. No se cobrarán más horas." -ForegroundColor Green
    }
    
    'status' {
        Write-Host "📊 Estado de la VM:" -ForegroundColor Cyan
        gcloud compute instances describe $VM_NAME --zone=$ZONE --format='table(name,status,machineType,networkInterfaces[0].accessConfigs[0].natIP:label=EXTERNAL_IP)'
    }
    
    'ssh' {
        Write-Host "🔌 Conectando por SSH..." -ForegroundColor Cyan
        gcloud compute ssh $VM_NAME --zone=$ZONE
    }
    
    'logs' {
        Write-Host "📋 Mostrando logs del servicio..." -ForegroundColor Cyan
        gcloud compute ssh $VM_NAME --zone=$ZONE --command='sudo journalctl -u exercise-api -n 50 -f'
    }
    
    'ip' {
        $IP = gcloud compute instances describe $VM_NAME --zone=$ZONE --format='get(networkInterfaces[0].accessConfigs[0].natIP)'
        Write-Host "📍 IP Pública: $IP" -ForegroundColor Cyan
        Write-Host "🌐 API URL: http://$IP/" -ForegroundColor Green
        Write-Host "🔌 WebSocket Pushup: ws://$IP/ws/pushup" -ForegroundColor Green
        Write-Host "🔌 WebSocket Squat: ws://$IP/ws/squat" -ForegroundColor Green
        Write-Host "🔌 WebSocket Plank: ws://$IP/ws/plank" -ForegroundColor Green
    }
    
    'delete' {
        Write-Host "⚠️  ADVERTENCIA: Esto eliminará la VM permanentemente." -ForegroundColor Red
        $confirm = Read-Host "¿Estás seguro? (si/no)"
        if ($confirm -eq "si") {
            gcloud compute instances delete $VM_NAME --zone=$ZONE
            Write-Host "✅ VM eliminada." -ForegroundColor Green
        } else {
            Write-Host "❌ Cancelado." -ForegroundColor Yellow
        }
    }
}
