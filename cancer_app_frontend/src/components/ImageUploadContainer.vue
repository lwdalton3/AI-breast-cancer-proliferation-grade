<template>
  <div>
    <v-card flat max-width="500" width="500" v-if="!waitingForPrediction && !responseImage">
      <file-pond
          label-idle="Drop images here or <span class='filepond--label-action'>Browse</span>"
          allow-multiple="true"
          accepted-file-types="image/jpeg, image/png"
          max-files="1"
          server="/backend/upload"
          v-bind:files="imagesToUpload"
          @processfiles="waitForPrediction"
          />
    </v-card>
    <v-container v-if="waitingForPrediction">
      <v-row>
        <v-col class="d-flex justify-center" cols="12">
          <v-progress-circular
              :size="100"
              :width="7"
              color="primary"
              indeterminate
          />
        </v-col>
      </v-row>
      <v-row>
        <v-col cols="12">
          Waiting for prediction
        </v-col>
      </v-row>
    </v-container>
    <v-container v-if="!waitingForPrediction && !!responseImage">
      <v-card flat class="grey lighten-4">
        <v-card-title>
          Prediction Results
        </v-card-title>
        <v-row>
          <v-col class="d-flex justify-center align-center">
            High: {{ predictionData.high }}
          </v-col>
          <v-col class="d-flex justify-center">
            Low: {{ predictionData.low }}
          </v-col>
          <v-col class="d-flex justify-center">
            Stroma: {{ predictionData.stroma }}
          </v-col>
        </v-row>
        <v-row>
          <v-col>
            <v-img
                :src="responseImage"
                max-height="1000"
                max-width="1000"
                />
          </v-col>
        </v-row>
      </v-card>
    </v-container>
  </div>
</template>

<script>
  // Import Vue FilePond
  import vueFilePond from 'vue-filepond'

  // Import FilePond styles
  import 'filepond/dist/filepond.min.css'

  // Import FilePond plugins
  // Please note that you need to install these plugins separately

  // Import image preview plugin styles
  import 'filepond-plugin-image-preview/dist/filepond-plugin-image-preview.min.css'

  // Import image preview and file type validation plugins
  import FilePondPluginFileValidateType from 'filepond-plugin-file-validate-type'
  import FilePondPluginImagePreview from 'filepond-plugin-image-preview'

  // Create component
  const FilePond = vueFilePond(
          FilePondPluginFileValidateType,
          FilePondPluginImagePreview
  )
  export default {
    name: 'ImageUploadContainer',
    data: function() {
      return {
        imagesToUpload: [],
        waitingForPrediction: false,
        predictionData: null,
        responseImage: null,
        checkIfRecogDoneTimer: null
      }
    },

    components: {
      FilePond
    },

    methods: {
      waitForPrediction() {
        // Activate circular loader
        this.waitingForPrediction = true

        // Activate timer which periodically checks if image recog is over
        this.checkIfRecogDoneTimer = setInterval(this.checkIfRecogDone, 3000)
      },

      checkIfRecogDone() {

        fetch('/backend/prediction')
          .then(response => response.json())
          .then(data => {
            if (data.status === 'Prediction complete.') {
              clearTimeout(this.checkIfRecogDoneTimer)
              this.predictionData = data.data
              fetch('/backend/download')
                .then(response => response.blob())
                .then(imageBlob => {
                  // Then create a local URL for that image and print it
                  const imageObjectURL = URL.createObjectURL(imageBlob)
                  this.responseImage = imageObjectURL
                  this.waitingForPrediction = false
                })
            }
          })
      }
    }
  }
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped></style>
