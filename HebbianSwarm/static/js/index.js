window.HELP_IMPROVE_VIDEOJS = false;


$(document).ready(function() {
    // Check for click events on the navbar burger icon

    var options = {
			slidesToScroll: 1,
			slidesToShow: 1,
			loop: true,
			infinite: true,
			autoplay: true,
			autoplaySpeed: 5983*3,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);
	
    bulmaSlider.attach();

})
bulmaCarousel.attach('#results-carousel', {
  slidesToScroll: 1,
  slidesToShow: 1,
  loop: true,
  autoplay: false,
  pauseOnHover: true,
  navigation: true,
  pagination: true,
  swipeable: true,
});

