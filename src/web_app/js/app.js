const port = process.env.PORT || 3001
const express = require('express')
const path= require('path')
const app = express()

const basePath = path.join(__dirname, '../public')
app.use(express.static(basePath))
//console.log(path.join(__dirname, '../public'))

app.listen(port, () => {
    console.log('Serveur sur le port : '+port )
})



