<template>
  <v-card flat max-width="500" width="500">
    <file-pond
        label-idle="Drop images here or <span class='filepond--label-action'>Browse</span>"
        allow-multiple="true"
        accepted-file-types="image/jpeg, image/png"
        max-files="1"
        server="/backend/upload"
        v-bind:files="imagesToUpload"
        />
  </v-card>
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
      return { imagesToUpload: [] }
    },

    components: {
      FilePond
    }
  }
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped></style>
