<template>
  <v-container fill-height>
    <v-row>
      <v-col class="d-flex justify-center">
        <ImageUploadContainer />
      </v-col>
    </v-row>
  </v-container>
</template>

<script>
  import ImageUploadContainer from '@/components/ImageUploadContainer'

  export default {
    name: 'HomeView',

    components: {
      ImageUploadContainer,
    }
  }
</script>
